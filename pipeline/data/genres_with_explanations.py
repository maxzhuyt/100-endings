"""
Story Taxonomy based on Eric Williams' "The Screenwriters Taxonomy".
Includes detailed definitions for Level 1 through Level 7.
"""

# LEVEL 1: TYPE
TYPES = {
    "Drama": [
        "Dark Drama: Dramas dealing with intensely serious issues.",
        "Docudrama: Fictionalized adaptation of real-life events where the general facts are more or less true.",
        "Docufiction: Combines non-fiction and fiction, intermingling actual historical records or facts with recreated scenes.",
        "Dramedy: A serious story containing characters or scenes inherently humorous to the reader.",
        "Hyperdrama: Exaggerates characters and situations to the point of becoming fable, legend, or fairy tale.",
        "Light Drama: Lighthearted stories that are, nevertheless, serious in nature.",
        "Satire: Uses irony or exaggeration to expose faults in society or individuals.",
        "Straight Drama: Broad category for stories that do not attempt a specific approach but consider drama as a lack of comedic techniques.",
        "Tragedy: Stories that explore human suffering and end in devastation.",
        "Tragicomedy: Explores human suffering but with an uplifting ending or enough comedic elements to keep the reader laughing."
    ],
    "Comedy": [
        "Bathroom Comedy: Indecent comedy, often about sex and bodily functions, containing a healthy dose of profanity.",
        "Comedy of Ideas: Explores serious ideas like religion or politics where characters represent divergent worldviews.",
        "Comedy of Manners: Emphasizes verbal gymnastics, clever insults, and witticisms above all else.",
        "Dark or Black Comedy: Humor that makes light of subject matter typically considered taboo.",
        "Farce: Exaggerates situations beyond the realm of possibility to make them entertaining.",
        "Observational Humor: Finds humor in the common practices of everyday life.",
        "Parody (or Spoof): Humor based on imitating and exaggerating a preexisting work to the point of mockery.",
        "Sex Comedy: Humor primarily derived from sexual situations and desire.",
        "Situational Comedy: Humor derived from exposing stock characters to different situations.",
        "Straight Comedy: Uses comedy for comedy’s sake without a specific approach.",
        "Slapstick Comedy: Humor that uses physical clumsiness, falling, and cartoon-like violence as the core element.",
        "Surreal Comedy: Storytelling that includes illogical behavior, bizarre juxtapositions, and absurd situations."
    ]
}

# LEVEL 2: SUPER GENRE
SUPER_GENRES = [
    "Action: Visceral stories featuring a resourceful hero against a single-minded villain, often involving intense physical conflict or chases.",
    "Crime: Explores truth, justice, and freedom containing the fundamental dichotomy of Criminal versus Lawman.",
    "Fantasy: Creates fantastical representations of humans’ subconscious fears and desires.",
    "Horror: An innocent victim is confronted by an overpowering, evil entity whose sole intention is to kill or corrupt.",
    "Life (Day-in-the-Life or Slice-of-Life): Takes small events in a person’s life and raises their level of importance.",
    "Romance: Focuses on 'special places' representing love in contrast to the mundane world.",
    "Science Fiction: The protagonist faces something unknown or incomprehensible with the potential to change the future.",
    "Sports: An underdog story typically focusing on 'Our Team' versus 'Their Team'.",
    "Thriller: Explores the ideas of hope and fear, constantly tearing the unwitting hero between these two extremes.",
    "War: A small group of isolated individuals get killed by an outside force until a final battle.",
    "Western: Explores the idea of order versus chaos in a frontier or untamed setting."
]

# LEVEL 3 & 4: MACROGENRES AND MICROGENRES
MACRO_MICRO_GENRES = {
    "Addiction: Protagonist struggles to overcome an addiction or interacts with someone who is.": [
        "Actions: Behaviors such as hoarding, eating disorders, self-inflicted harm, or kleptomania.",
        "Alcohol/Drugs/Gambling: Specific substances or activities the character is addicted to."
    ],
    "Adventure: Focuses on exotic locations and unusual characters the protagonist encounters.": [
        "Battle/Exploration/Piracy/Quest/Rebellion/Space: Specific contexts for the adventure."
    ],
    "Alien Invasion: Centers around an Us-versus-Them dichotomy involving extraterrestrials.": [
        "Earth-Centric: Heroes try to save the planet, country, or local community.",
        "First Contact: Highlights the awe of realizing humans are not the only living creatures.",
        "Space-Centric: Focuses on the adventure of battle elsewhere."
    ],
    "Apocalyptic: Defined by where the story takes place in proximity to the end of the world.": [
        "Actively Within/Postapocalyptic/Preapocalyptic: Timing of the story relative to the apocalyptic event."
    ],
    "Artificial Intelligence: Explores human trust in machines and what makes us human.": [
        "Androids: Robots that appear to be human.",
        "Cyborgs: Beings that are both mechanical and organic.",
        "Robots: Beings that are 100% mechanical."
    ],
    "Biography/Character Study: Can be parsed into micros based on the subject's status or the time period.": [
        "Entertainment/Everyman/Group/Newsworthy/Political/Rich and Famous/Rags-to-Riches: The nature of the subject being profiled.",
        "Temporal: Whether the story is modern day, historical, or a period piece."
    ],
    "Bro- or Womance: Tells the story of a strong platonic relationship between two people of the same sex.": [
        "Stronger Bond/Traditional/Tragedy: The nature or outcome of the relationship."
    ],
    "Demonic: Evil is personified and the protagonist battles it physically or spiritually.": [
        "Devil/Occult: Protagonist battles the Devil or occult forces.",
        "Possession: Demonic influence made physical through a human.",
        "Witchcraft: Evil attacks in the form of spell-casting figures."
    ],
    "Disaster: Protagonists realize that without action, their world is going to end.": [
        "Global/Local/Personal: The scope of the disaster."
    ],
    "Disease/Disability: Protagonists realize life as they know it has ended or is about to end due to health issues.": [
        "Personal: Struggles like Alzheimer's or physical disabilities.",
        "Societal: Threats like a global pandemic."
    ],
    "Epic/Saga: Long, detailed stories or spanning narratives of heroic adventures.": [
        "Discovery/Exploration/Event-Specific/Historical/Person-Specific: The focus of the epic narrative."
    ],
    "Erotica: Uses sexual encounters as a central storytelling construct.": [
        "Fetish: Explores specific sexual desires.",
        "Mainstream: Sexual encounters drive the main characters."
    ],
    "Escape: The protagonist must be trapped in a place that is well guarded and fortified.": [
        "Impossible: Focuses on a target that seems unreachable.",
        "Nonprison: Escaping from situations like abusive families or wilderness wrecks.",
        "Tale: Interested in telling the story of the escapee rather than the mechanics.",
        "Traditional: Traditional prison escape stories."
    ],
    "Family: Revolves around how the family as a whole reacts to specific events.": [
        "Family Bond: Family bonds during difficult emotional times.",
        "Family Feud: Disparate members argue and eventually resolve differences.",
        "Family Loss: Comments on how a family deals with tragedy.",
        "Family Rift: Explores dynamics when an irreconcilable difference occurs."
    ],
    "Gangs/Punks/Brothers in Arms: Focuses on the group dynamic over the story of an individual.": [
        "Ideology-Specific: Characters aligned by beliefs like politics or counter-culture.",
        "Interest-Specific: Characters aligned by hobbies or subcultures.",
        "Location-Specific: Characters aligned by neighborhood or school."
    ],
    "Gangster: Explores characters within the strict hierarchy of organized crime.": [
        "Antihero POV: Told from the criminal's perspective.",
        "Authority POV: Told from the law enforcement perspective."
    ],
    "Ghost/Spirits/Angels: Uses spectral characters in a metaphoric way beyond simple horror.": [
        "Across the Divide/Emotional/Mystic/Reincarnation: The function of the spirit in the story."
    ],
    "Heist/Caper: A band of 'heroes' systematically solves problems to attempt a robbery.": [
        "Impossible: Focuses on a target that seems unreachable.",
        "Procedural: Engages the reader as members of the team explaining specifics.",
        "Tale: Interested in telling the story of the thief rather than the mechanics."
    ],
    "Historical: Provides the reader new insight into their own understanding of history.": [
        "Alternate: Parallels fictional elements with historically accurate events to create a 'what if' scenario.",
        "Modern/Period: Context for the historical insight."
    ],
    "Holiday: Meant to reinforce the theme of a specific holiday.": [
        "Event: Focuses on the celebration itself.",
        "Family: Involves family conflicts resolved by the holiday spirit.",
        "Origin: Reminds the reader of age-old tales about the holiday."
    ],
    "Identity: Highly personalized tales about someone struggling to find out who they are.": [
        "Gender/Sexuality/Humanity/Loss of.../Racial/Religious: The specific aspect of identity being explored."
    ],
    "Killing: Death is the central issue; someone has been killed or is going to be killed.": [
        "Reflection: Focuses on those who have already killed reflecting on the act.",
        "Serial: Focuses on the killer rather than the victims.",
        "Slasher: A group is systematically killed by a deranged killer.",
        "Stalker: Focuses on the events leading up to a killing.",
        "Unintended: Dealing with accidental death."
    ],
    "Law Enforcement: Focuses on protecting the greater good rather than just solving a mystery.": [
        "Community/Personal Crime/Process/Rescue/True Crime/Undercover: The specific focus of the enforcement activity."
    ],
    "Legal: Focuses on the legal battle at hand.": [
        "Courtroom: Teams vying to win a case.",
        "Investigation: The reader decides guilt or innocence alongside the detective.",
        "Tales of the System: Focuses on how the legal process works.",
        "Underdog/Whistleblower: Standing up against powerful institutions."
    ],
    "Love: Characters make every major decision based upon love.": [
        "In Disguise: Appears to be one type of relationship but reflects another.",
        "Nonconventional: Involves participants who are not human.",
        "Obsession: Implies an unhealthy relationship where one is preoccupied by the other.",
        "Traditional: Two people who feel the same way about each other.",
        "Unrequited: The intensity of one character's love is not reciprocated."
    ],
    "Magical: Mysterious forces play a central role in the plot or outcome.": [
        "Locational: Character is transported to a magical land.",
        "Vocational: Characters are magicians or magic-users by trade.",
        "Wonderment: Magic serves to create awe."
    ],
    "Martial Arts: Delineated by the form of martial art or the philosophical approach.": [
        "Form: Highlights specific arts like karate or kung fu.",
        "Philosophy: Highlights rules of honor (samurai) or irregular warfare (ninja)."
    ],
    "Medical: Highlights the lives and acts of medical professionals.": [
        "Biography/Nonhospital/Seemingly Impossible/Tale: The setting or nature of the medical challenge."
    ],
    "Military: Highlights the life and acts of military professionals.": [
        "Biographical/Historical/Mission/Tale: The scope of the military story."
    ],
    "Mission: Explains details to the reader so they can enjoy the improvisation when things go wrong.": [
        "Impossible: Predicated on the assumption the mission will fail.",
        "Tale: A risky yet logistically possible operation."
    ],
    "Monster: Represents our inner fears broken down into psychological groups.": [
        "Alien/Creature/Mutant/Mythic/Scientific/Swarms: The specific type of entity causing fear."
    ],
    "Mystery/Detective: Explores the clues and motives that reveal a crime and the person uncovering them.": [
        "Crime-Specific: Assumes the POV of the detective regarding a specific crime.",
        "Hard-Boiled: Exploration of a cynical, emotionless character in criminal settings.",
        "Pulp: Hard-boiled world where everyone is cynical.",
        "Whodunit: Detective tries to figure out who committed an unsolved crime."
    ],
    "Political: Addresses the inner workings of governmental decision making.": [
        "Contemporary/Historical/International/Local/National: The scope and timing of the political story."
    ],
    "Procedural: Highlights a specific professional process, minimizing interpersonal behavior.": [
        "Artistic/Investigative/Legal/Medical/Political/Scientific: The profession being detailed."
    ],
    "Protection: Focuses as much on the people being protected as on the protector.": [
        "Babysitter/Bodyguard/Family/Legal: The relationship between protector and protected."
    ],
    "Psychological: Focuses on the mental and emotional degradation of the protagonist.": [
        "Conspiracy: Protagonist is in the center of a plot they don't understand.",
        "Evil/Mistaken Identity/What If?/Worst Case: The source of the psychological torment."
    ],
    "Religious: Explores the spiritual awakening of a character.": [
        "Biographical/Religious Text/Spirituality: Source of the religious narrative."
    ],
    "Revenge/Justice: Focuses on a character's obsession with exacting retribution.": [
        "Legal/Psychological/Settle a Score/Vigilante: The method or arena of revenge."
    ],
    "Romantic Comedy: Takes a light and humorous touch to relationships where the reader knows the characters are perfect for each other.": [
        "Adult/Child/Nonconventional/Remarriage/Teen: The nature of the participants."
    ],
    "School Stories: School life becomes its own universe with sharp dichotomies.": [
        "College/Elementary/High School/Specific School: The educational setting."
    ],
    "Science Fantasy: Revels in creating stories that are scientifically impossible.": [
        "Discovery/Dying Earth/ET Relations/Mad Scientist/Space Opera/Sword & Planet: The specific fantasy element."
    ],
    "Showbiz/Artistry: Lingers on characters' societal practices within entertainment or artistic culture.": [
        "Biography/Creative Process/Culture: The focus of the artistic story."
    ],
    "Slasher: Highlights violence with sharp objects where skin is slashed.": [
        "Splatter: Focuses on graphic, visceral details of violence.",
        "Survival: Dramatizes the lone survivor's escape.",
        "Teen: Plays up teenage relationships."
    ],
    "Spy/Espionage: Uses covert means to obtain secret information for combative purposes.": [
        "Corporate/Military/Political/Secrets: The arena of espionage."
    ],
    "Superhero: Divided by the scope of the hero's battle.": [
        "Origin Story: Explains how a superhero obtained and first started using powers.",
        "Personal Battles: Villain's acts affect those the hero cares about.",
        "Save the World: Hero tries to save humanity."
    ],
    "Superpowers: Explores the lives of ordinary people with extraordinary abilities.": [
        "Intelligence/Life Extension/Mental/Physical: The nature of the ability."
    ],
    "Survival: Questions whether the protagonist will live until the end of the story.": [
        "Captivity: Surviving by adapting to the environment.",
        "Experiences: Surviving life-threatening events within society.",
        "Nature/Outdoor: Personal efforts to stay alive in the wild.",
        "Rescue: Character can only survive with help from others."
    ],
    "Terror: Revels in the psychological buildup to an impending horrific incident.": [
        "AI/ET/Man-Made/Natural/Paranormal/Stalker/Survival: The source of the terror."
    ],
    "Time Travel: Dramatizes paradoxes or time manipulation.": [
        "Discovered: Phenomenon is seen as a superpower or curse.",
        "Exists: Travel is common knowledge and rarely questioned.",
        "Secret Invention: Paradoxes come into play as few understand the ramifications."
    ],
    "Workplace: Views characters in a two-dimensional way, only seeing them at work.": [
        "Boss’s Story/What It’s Like/Worker’s Story: The perspective of the workplace narrative."
    ]
}

# LEVEL 5: VOICE
VOICE = {
    "Narrative Structure": [
        "Linear Narrative: Each scene progresses forward in time.",
        "Nonlinear (Flashbacks): Takes the reader back in time for a flash of something that happened previously.",
        "Nonlinear (Two time periods, intercut): Two stories told simultaneously.",
        "Nonlinear (Parallel realities): Cuts between two time periods where only one is 'real'.",
        "Nonlinear (Repetition of events): Repeats the same set of events multiple times.",
        "Nonlinear (Time Travel): Scenes set in a nonlinear order to confuse and entertain.",
        "Nonlinear (Reverse Chronology): Scenes run backward through time."
    ],
    "Prose Style and Pacing": [
        "Mixing Formatting/Styles: Using different text styles, fonts, or stylistic shifts to enhance the story.",
        "Minimalist Prose: Portraying the story through sparse, unadorned language and subtext.",
        "Unembellished Narrative: Absence of poetic or flowery exposition; raw delivery through everyday language.",
        "Extended Scenes/Deep Introspection: Focus on continuous, real-time action or deep psychological exploration rather than structured plot movement.",
        "Static Perspective: Eschewing sweeping narrative shifts in favor of a grounded, localized viewpoint."
    ],
    "Literary Format": [
        "Flash Fiction/Vignette: Used for its brief, concentrated, and highly stylized quality.",
        "Prose Poetry: Blending poetic, lyrical elements with standard prose.",
        "Epistolary/Mixed Media: Telling the story through letters, diary entries, emails, or reports.",
        "Standard Prose: Traditional paragraph formatting and narrative flow."
    ],
    "Dialogue and Interiority": [
        "Lyrical/Poetic Monologue: Characters express innermost thoughts through highly stylized, rhythmic prose.",
        "Wordless/Action-Driven: Removing all spoken dialogue between characters, relying solely on action and observation.",
        "Stream of Consciousness: Allows the reader direct, unfiltered access to what characters are thinking."
    ],
    "Metafiction and Faux-Nonfiction": [
        "Breaking the Fourth Wall: Fictional characters or the narrator address the reader directly.",
        "Pseudo-Documentary (Found Footage): Characters compile a grassroots report, journal, or interview log.",
        "Pseudo-Documentary (Sham-Reality): Acting as if a scripted, fictional story were a true historical account.",
        "Pseudo-Documentary (Historical Figure): Uses known historical figures to hook people into a fictional story.",
        "Pseudo-Documentary (Satiric-Media): Uses journalistic approaches to satirize media itself.",
        "Pseudo-Documentary (Uchronia): Documents actions occurring in a parallel universe as if they were fact."
    ],
    "Eponymous Styles": [
        "Borgesian: Explores metafictional paradoxes, labyrinths, infinite loops, creative use of mathematical concepts, and the blurring of reality and illusion.",
        "Chekhovian: Prioritizes understated realism, mood, subtext, and subtle character epiphanies over traditional plot-driven action.",
        "Dickensian: Focuses on social realities, like impoverished but noble characters, exaggerated and satirical caricatures, and critiques of systemic injustice.",
        "Dostoevskian: Plunges deep into the psychological and spiritual torment, existential dread, and moral ambiguity of its characters.",
        "Faulknerian: Employs nonlinear timelines, fragmented prose, and multiple unreliable perspectives to explore the inescapable weight of history on individual lives and choices.",
        "Gogolian: Blends the mundane realities of everyday life with the grotesque, the absurd, and biting social satire.",
        "Joycean: Utilizes highly experimental language, wordplay, allusions, and stream of consciousness to reveal the profound within the mundane.",
        "Kafkaesque: Portrays characters trapped in nightmarish, illogical, and oppressive bureaucratic systems or surreal situations, where stories usually do not have an ending.",
        "Pinteresque: Uses mundane, ambiguous dialogue filled with pauses and silences to mask underlying power struggles and creeping menace.",
        "Proustian: Centers on involuntary memory, elastic time, and the sensory, personal exploration of the past through a deeply introspective perspective.",
        "Woolfian: Captures the fluid, associative, and subjective interior lives of characters in deeply impressionistic prose, weaving them into a highly interconnected social fabric."
    ]
}

# LEVEL 6: PATHWAY
PATHWAYS = [
    "The Defeated Underdogs (Noir/Neo-noir): Protagonist is rarely successful and spends the story in distrust or despair.",
    "The Defeated Underdogs (Tale of Madness): Hero battles their nemesis and their own insanity.",
    "The Defeated Underdogs (Rags to Riches to Rags): Hero succeeds but loses it all by the end.",
    "Melodrama (Defeated Underdog + Subverted): Focuses on how the main character endures a tragic hardship.",
    "Subverted Journeys (Chase/Hunt): Protagonist is always at the mercy of the antagonist and acting like prey.",
    "Subverted Journeys (Road Trip/Journey): Protagonist embarks on a journey but does not return to their normal world.",
    "Multiple Protagonists (Buddy Story): Twin protagonists share the same general goals but different driving forces.",
    "Multiple Protagonists (Screwball Comedy): Central male character is joined by a female character who challenges his masculinity.",
    "Multiple Protagonists (Assemble the Team): Protagonist rounds up a team of experts to accomplish a goal.",
    "Multiple Protagonists (Unlikely Ensemble): Characters are brought together by outside forces, not friendship.",
    "Multiple Protagonists (Reunion): Group rekindles relationships after being separated for years.",
    "Multiple Protagonists (Group Falls Apart): Unified group disappears or breaks loyalty one by one.",
    "All-Knowing Reader (Coming of Age): Reader is familiar with the lesson the protagonist is going to learn.",
    "All-Knowing Reader (Lost Innocence): Protagonist goes through something more extreme than most people experience.",
    "Noncharacter Antagonists (Fish Out of Water): The new environment is the problem, not a specific antagonist.",
    "Noncharacter Antagonists (Human vs. Nature): Protagonist is simply trying to survive natural conditions.",
    "Noncharacter Antagonists (Human vs. Self): The real battle is within the protagonist.",
    "Noncharacter Antagonists (Human vs. Society): Individual struggles against community expectations and norms.",
    "Noncharacter Antagonists (Human vs. Technology): Protagonist battles against a machine or technological ripples."
]

# LEVEL 7: POINT OF VIEW
POV = [
    "Authorial Omniscient: The narrator takes the reader on a tour of characters and events designed to intrigue, knowing all thoughts and outcomes.",
    "Primary Omniscient: The story is viewed through the main character’s personal lens, but the narrative can take the reader anywhere.",
    "Primary Limited: Limits the knowledge available to the reader strictly to what a single character knows and perceives (First-person or Third-person Limited).",
    "Secondary Limited: Story is about the primary character but told from the perspective of a secondary character (e.g., Watson narrating Sherlock Holmes).",
    "Secondary Omniscient: Secondary character reflects on life or has special insight causing a broader omniscient perspective."
]