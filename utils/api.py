import os
import time
import logging
import json
import requests
import random
import string
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

class APIClient:
    """
    Client for interacting with LLM API endpoints (OpenAI, Anthropic, OpenRouter, etc.).
    Mimics eqbench usage: we have 'test' vs 'judge' model_type references.
    """

    def __init__(self, model_type=None, request_timeout=240, max_retries=3, retry_delay=5):
        self.model_type = model_type or "default"

        if model_type == "test":
            self.api_key = os.getenv("TEST_API_KEY", os.getenv("OPENAI_API_KEY"))
            self.base_url = os.getenv("TEST_API_URL", os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
        elif model_type == "judge":
            self.api_key = os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY"))
            self.base_url = os.getenv("JUDGE_API_URL", os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", request_timeout))
        self.max_retries = int(os.getenv("MAX_RETRIES", max_retries))
        self.retry_delay = int(os.getenv("RETRY_DELAY", retry_delay))

        # Determine API provider for header/payload structure
        self.provider = "openai"  # Default
        if "anthropic.com" in self.base_url:
            self.provider = "anthropic"

        self.headers = self._get_headers()

        logging.debug(f"Initialized {self.model_type} API client. Provider: {self.provider}, URL: {self.base_url}")

    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.provider == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _prepare_anthropic_messages(self, messages: List[Dict[str, str]]) -> tuple:
        """Extract system prompt and format messages for Anthropic API."""
        system_prompt = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                # Anthropic requires alternating user/assistant roles
                if not user_messages or user_messages[-1]["role"] != msg["role"]:
                    user_messages.append(msg)
                else:
                    # Merge consecutive messages of same role
                    user_messages[-1]["content"] += "\n\n" + msg["content"]
        return system_prompt, user_messages

    def _extract_anthropic_content(self, data: Dict[str, Any]) -> str:
        """Extract text content from Anthropic API response."""
        if data.get("type") == "error":
            raise RuntimeError(f"Anthropic API Error: {data.get('error', {}).get('message')}")
        if isinstance(data.get("content"), list):
            text_block = next((block["text"] for block in data["content"] if block.get("type") == "text"), None)
            if text_block:
                return text_block
            return ""
        return data.get("completion", "")

    def generate(self, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 8096, include_seed=True, min_p = 0.1, system=None) -> str:
        """
        Generic chat-completion style call.  We allow an optional random seed block.
        """

        #max_tokens=28000

        messages = [{"role": "user", "content": prompt}]
        if system:
            messages = [{"role": "system", "content": system}] + messages

        # Optionally add random seed block as a system message for judging tasks.
        # This allows us to get variation between iterations without using temp > 0 which compromises judging performance.
        # The reason for doing this is to understand *judging* variance from the same inputs, i.e. when
        # using --redo-judging. In most use cases you won't need to worry about this and can leave it disabled.
        if False:
            if include_seed:
                seed_lines = [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=80)) for _ in range(5)
                ]
                random_seed_block = (
                    "<RANDOM SEED PLEASE IGNORE>\n" +
                    "\n".join(seed_lines) +
                    "\n</RANDOM SEED>"
                )
                messages = [{"role": "system", "content": random_seed_block}] + messages

        for attempt in range(self.max_retries):
            response = {}
            try:
                # Build payload based on provider
                if self.provider == "anthropic":
                    system_prompt, user_messages = self._prepare_anthropic_messages(messages)
                    payload = {
                        "model": model,
                        "messages": user_messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    if system_prompt:
                        payload["system"] = system_prompt
                    # Disable reasoning/thinking for Anthropic models
                    payload["thinking"] = {"type": "disabled"}
                    if model == "claude-opus-4-5-20251101":
                        print('enable thinking for opus 4.5')
                        payload["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": 4096 
                        }
                else:
                    # OpenAI format
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                    if min_p != None and model != 'o3' and self.base_url != 'https://api.openai.com/v1/chat/completions':
                        # Only use min_p for the test model (not judge).
                        # If your test model doesn't support min_p, you may need to
                        # disable this here. Alternatively you could use openrouter
                        # which will automatically omit unsupported params.
                        payload['min_p'] = min_p
                    if self.base_url == 'https://api.openai.com/v1/chat/completions':
                        try:
                            del payload['min_p']
                        except:
                            pass

                    if model == 'o3':
                        # o3 has special reqs via the openai api
                        del payload['max_tokens']
                        payload['max_completion_tokens'] = max_tokens
                        payload['temperature'] = 1
                    if model in ['gpt-5-2025-08-07', 'gpt-5-mini-2025-08-07', 'gpt-5-nano-2025-08-07']:
                        payload['reasoning_effort']="minimal"
                        del payload['max_tokens']
                        payload['max_completion_tokens'] = max_tokens
                        payload['temperature'] = 1

                    if 'openrouter' in self.base_url and model == "openai/gpt-5.2":
                        print('gpt5.2 no reasoning')
                        payload["reasoning"] = {"effort": "none"}

                    if model in ['gpt-5-chat-latest']:
                        del payload['max_tokens']
                        payload['max_completion_tokens'] = max_tokens
                        payload['temperature'] = 1

                    if model == 'anthropic/clause-sonnet-4':
                        payload['reasoning'] = {
                            "max_tokens": 0
                        }
                    if 'qwen3' in model.lower() and model != 'qwen/qwen3-next-80b-a3b-instruct':
                        # optionally disable thinking for qwen3 models
                        print('/no_think')
                        system_msg = [{"role": "system", "content": "/no_think"}]
                        messages = system_msg + messages
                        payload['messages'] = messages

                    if model == 'anthropic/claude-sonnet-4.5':
                        payload["max_tokens"] = 16000

                    if model == "hermes4":
                        print('injecting reasoning sys prompt')
                        reasoning_sys = "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."
                        system_msg = [{"role": "system", "content": reasoning_sys}]
                        messages = system_msg + messages
                        payload["messages"] = messages
                        payload["max_tokens"] = 12000

                    if model == "moonshotai/kimi-k2-thinking" or model == "kimi-k2-thinking":
                        payload["max_tokens"] = 32000

                    # Disable reasoning for Anthropic models via OpenRouter
                    if 'openrouter' in self.base_url and model.startswith('anthropic'):
                        payload["reasoning"] = {"max_tokens": 0}

                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                data = response.json()

                # Extract content based on provider
                if self.provider == "anthropic":
                    content = self._extract_anthropic_content(data)
                else:
                    content = data["choices"][0]["message"]["content"]

                # strip out any <think> blocks if the model yields that
                if '<think>' in content and "</think>" in content:
                    post_think = content.find('</think>') + len("</think>")
                    content = content[post_think:]
                if '<thinking>' in content and "</thinking>" in content:
                    post_think = content.find('</thinking>') + len("</thinking>")
                    content = content[post_think:]
                if '<reasoning>' in content and "</reasoning>" in content:
                    post_think = content.find('</reasoning>') + len("</reasoning>")
                    content = content[post_think:]
                return content
            except requests.exceptions.Timeout:
                logging.warning(f"Request timed out on attempt {attempt+1}/{self.max_retries}")
            except requests.exceptions.HTTPError as e:
                logging.error(e)
                if response:
                    print(response)
                if e.response.status_code == 429:
                    logging.warning("Rate limit. Backing off.")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
            except Exception as e:
                logging.error(e)
                logging.warning(f"Error on attempt {attempt+1}/{self.max_retries}: {str(e)}")

            time.sleep(self.retry_delay)

        raise RuntimeError(f"Failed to generate text after {self.max_retries} attempts")




