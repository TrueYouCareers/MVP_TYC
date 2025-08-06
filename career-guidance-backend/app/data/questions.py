"""
Questionnaire data for different education levels
"""

# QUESTIONNAIRE_DATA = {
#     "10th": {
#         "q1": {
#             "text": "What activity excites you the most?",
#             "options": {
#                 "a": "Conducting science experiments",
#                 "b": "Debating or public speaking",
#                 "c": "Planning and budgeting money",
#                 "d": "Writing stories or articles"
#             },
#             "category": "interest_discovery"
#         },
#         "q2": {
#             "text": "Which of these best describes you?",
#             "options": {
#                 "a": "Analytical & Curious",
#                 "b": "Expressive & Logical",
#                 "c": "Structured & Practical",
#                 "d": "Empathetic & Creative"
#             },
#             "category": "personality_type"
#         },
#         "q3": {
#             "text": "If given a project, what role do you enjoy most?",
#             "options": {
#                 "a": "Researcher",
#                 "b": "Leader/Speaker",
#                 "c": "Organizer",
#                 "d": "Designer/Creator"
#             },
#             "category": "work_preference"
#         }
#     },
#     "12th": {
#         "q1": {
#             "text": "Which subjects do you find most engaging?",
#             "options": {
#                 "a": "Physics, Chemistry, Mathematics",
#                 "b": "History, Political Science, Economics",
#                 "c": "Business Studies, Accountancy, Economics",
#                 "d": "Literature, Psychology, Sociology"
#             },
#             "category": "academic_interest"
#         }
#         # Add more 12th grade questions
#     },
#     "graduate": {
#         "q1": {
#             "text": "What type of work environment appeals to you?",
#             "options": {
#                 "a": "Research lab or technical facility",
#                 "b": "Corporate office with team collaboration",
#                 "c": "Creative studio or flexible workspace",
#                 "d": "Field work or community interaction"
#             },
#             "category": "work_environment"
#         }
#         # Add more graduate questions
#     }
# }

QUESTIONNAIRE_DATA = {
    "10th": {
        "q1": {
            "text": "Which subjects do you feel most confident in, without needing much help?",
            "options": {
                "a": "Mathematics",
                "b": "Physics / Chemistry",
                "c": "Biology / Environmental Science",
                "d": "History / Political Science / Geography / Current Affairs",
                "e": "Business Studies / Accounts / Economics",
                "f": "Computers / Tech / Coding"
            },
            "category": "academic_strengths_subject"
        },
        "q2": {
            "text": "Which of these areas do you enjoy the most or feel naturally drawn to?",
            "options": {
                "a": "Solving physics, chemistry, or math-based problems (PCM)",
                "b": "Learning about the human body, diseases, environment, and biology (PCB)",
                "c": "Learning how businesses and entrepreneurs work, and how finance creates impact (Commerce)",
                "d": "Exploring history, politics, society, and great leaders (Arts)",
                "e": "I enjoy multiple things and haven't figured it out yet"
            },
            "category": "stream_inclination"
        },
        "q3": {
            "text": "How would you rate your recent marks in the following subjects?",
            "options": {
                "Math": ["40–50%", "50–60%", "60–70%", "70–80%", "80–90%", "90%+"],
                "Science": ["40–50%", "50–60%", "60–70%", "70–80%", "80–90%", "90%+"],
                "Social Science": ["40–50%", "50–60%", "60–70%", "70–80%", "80–90%", "90%+"],
                "English/Hindi": ["40–50%", "50–60%", "60–70%", "70–80%", "80–90%", "90%+"],
                "Computer IT": ["40–50%", "50–60%", "60–70%", "70–80%", "80–90%", "90%+"]
            },
            "category": "academic_strengths_marks"
        },
        "q4": {
            "text": "At 30 years of age, what kind of profession do you see yourself in?",
            "options": {
                "a": "Software Developer / Engineer",
                "b": "Finance Professional / CA / Analyst",
                "c": "Doctor / Healthcare",
                "d": "Sportsperson",
                "e": "Government Officer (IAS, IPS)",
                "f": "Lawyer / Legal Expert",
                "g": "Teacher / Educator",
                "h": "Business Owner / Entrepreneur",
                "i": "Scientist / Researcher",
                "j": "Social Worker / NGO"
            },
            "category": "future_profession_goal"
        },
        "q5": {
            "text": "Which of these interests or hobbies reflect you the most?",
            "options": {
                "a": "Writing stories, poems, or expressing your thoughts clearly",
                "b": "Using design/editing tools (like Canva, Figma, etc.) to create digital content",
                "c": "Learning how science and math concepts work in real life",
                "d": "Speaking confidently, debating, or sharing opinions",
                "e": "Learning how businesses run and how entrepreneurs earn, grow, and impact society"
            },
            "category": "natural_hobbies"
        },
        "q6": {
            "text": "What kind of workplace excites you the most?",
            "options": {
                "a": "Government office setup",
                "b": "Tech/startup office space",
                "c": "Hospital or Research Lab",
                "d": "School or College",
                "e": "Own business / Remote work"
            },
            "category": "exciting_workplace"
        },
        "q7": {
            "text": "How do you usually spend your free time — when no one is forcing you?",
            "options": {
                "a": "Reading biographies, motivational or spiritual content",
                "b": "Watching or reading about history, political events, or current affairs",
                "c": "Exploring new tech, coding, or trying AI tools and new apps",
                "d": "Playing indoor or outdoor sports"
            },
            "category": "free_time_activity"
        },
        "q8": {
            "text": "Do you feel influenced by anyone while choosing your stream or future path?",
            "options": {
                "a": "My parents/family",
                "b": "My friends or classmates",
                "c": "Career trends/social media",
                "d": "I choose based on my own interest",
                "e": "I haven’t figured it out yet"
            },
            "category": "career_influence"
        },
        "q9": {
            "text": "What kind of school activity would you enjoy most or love to try?",
            "options": {
                "a": "Debates, quizzes, or political discussions",
                "b": "Tech fairs, science exhibitions, coding contests",
                "c": "Group projects or team-based tasks",
                "d": "Creative writing or speech competitions",
                "e": "Business idea pitches or finance-related activities",
                "f": "I haven’t figured it out yet"
            },
            "category": "activity_preference"
        },
        "q10": {
            "text": "If you had to create a YouTube channel or Instagram page, what would you enjoy making content about?",
            "options": {
                "a": "Explaining a science or math topic you learned",
                "b": "Sharing knowledge about human biology or health",
                "c": "Talking about history, politics, or current affairs",
                "d": "Reviewing apps, talking about tech or coding",
                "e": "Creating content about sports or fitness",
                "f": "Talking about inspiring figures like Ratan Tata, Gandhi, Swami Vivekananda, or Subhash Bose"
            },
            "category": "youtube_channel_interest"
        },
        "q11": {
            "text": "What do you enjoy doing so much that you lose track of time?",
            "options": {
                "a": "Learning how physics, chemistry, or math concepts work in real life",
                "b": "Learning about the human body, diseases, nature, and how living beings function",
                "c": "Exploring technology, coding, apps, or AI",
                "d": "Teaching, writing, or sharing thoughts with peers",
                "e": "Learning how businesses work and how entrepreneurs build things"
            },
            "category": "flow_activity"
        },
        "q12": {
            "text": "Which of the following describes your personality best?",
            "options": {
                "a": "I like to lead, speak up, and make an impact",
                "b": "I prefer calm focus and enjoy working deeply",
                "c": "I enjoy helping, guiding, or supporting people",
                "d": "I enjoy creating useful things and solving problems",
                "e": "I’m a balanced person — calm yet a leader, focused yet a team player"
            },
            "category": "personality_type"
        },
        "q13": {
            "text": "Why do most students around you choose their career paths?",
            "options": {
                "a": "Because of family pressure",
                "b": "For a high-paying or trending career",
                "c": "Because of friends",
                "d": "Due to their true interest",
                "e": "They’re still figuring it out"
            },
            "category": "career_reason_observed"
        },
        "q14": {
            "text": "Are you confident about the stream/career you want to choose?",
            "options": {
                "a": "Yes, I’m confident and clear",
                "b": "No, I feel unsure or confused"
            },
            "category": "confidence_stream_choice"
        },
        "q14_1": {
            "text": "If 'No', what makes you feel unsure? (Select all that apply)",
            "options": {
                "a": "My parents want something else",
                "b": "Friends are influencing my thinking",
                "c": "Too many options to choose from",
                "d": "I don’t know what I’m good at",
                "e": "I haven’t explored or thought about it yet",
                "f": "I need proper guidance or help"
            },
            "category": "unsure_reason_multiple"
        },
        "q15": {
            "text": "Which statement feels the most like 'you'? Be honest!",
            "options": {
                "a": "I love to lead, take charge, and speak up. I want to make an impact in the world.",
                "b": "I prefer calm focus. I enjoy working quietly and solving things in depth.",
                "c": "Helping people, mentoring, and guiding makes me happy. I enjoy being supportive.",
                "d": "I like creating something useful, like a project, a tool, or even a small startup."
            },
            "category": "self_statement"
        }
    },
    "12th": {
        "q1": {
            "text": "Which stream and subjects did you choose after 10th class?",
            "options": {
                "a": "Science",
                "b": "Commerce",
                "c": "Arts"
            },
            "category": "stream_chosen"
        },
        "q2": {
            "text": "What was the main reason you chose these subjects?",
            "options": {
                "a": "Genuine interest",
                "b": "Good marks in 10th",
                "c": "Parental or societal pressure",
                "d": "Peer influence",
                "e": "Seemed like a safe option",
                "f": "Random or unclear",
                "g": "Other (specify)"
            },
            "category": "reason_for_subject_choice"
        },
        "q3": {
            "text": "Are you confident about the subjects you chose?",
            "options": {
                "a": "Very confident",
                "b": "Somewhat confident",
                "c": "Not confident at all"
            },
            "category": "subject_confidence_level"
        },
        "q4": {
            "text": "If you had a free hand, which subjects would you love to study? (Multiple select)",
            "options": {
                "a": "Physics / Chemistry",
                "b": "Biology",
                "c": "Mathematics",
                "d": "Business Studies",
                "e": "Accountancy",
                "f": "Economics",
                "g": "History / Polity / Sociology / Current Affairs",
                "h": "Psychology",
                "i": "Geography",
                "j": "Others"
            },
            "category": "dream_subjects_multiple"
        },
        "q5": {
            "text": "If you imagine yourself 5–10 years from now, which profession excites you most?",
            "options": {
                "a": "Doctor",
                "b": "Engineer",
                "c": "Lawyer",
                "d": "Entrepreneur / Architect",
                "e": "Teacher / Professor",
                "f": "IAS / IPS / IFS",
                "g": "Creative Professional",
                "h": "Data Analyst",
                "i": "Research Scientist",
                "j": "CA / Financial Analyst",
                "k": "Journalist / Content Creator",
                "l": "Psychologist / Counselor",
                "m": "NGO/Social Work",
                "n": "Athlete",
                "o": "Army Officer",
                "p": "Other"
            },
            "category": "future_profession_excite"
        },
        "q6": {
            "text": "Which subjects do you truly enjoy studying? (Multiple select)",
            "options": {
                "a": "Physics, Chemistry, Biology",
                "b": "Math",
                "c": "Business Studies / Accountancy / Economics",
                "d": "History / Polity / Sociology",
                "e": "Psychology / Geography / Fine Arts",
                "f": "Other"
            },
            "category": "enjoyed_subjects_multiple"
        },
        "q7": {
            "text": "Which activities do you naturally enjoy the most? (Choose up to 3)",
            "options": {
                "a": "Public speaking",
                "b": "Solving logical puzzles",
                "c": "Creativity / Designing",
                "d": "Leading teams",
                "e": "Data analysis",
                "f": "Writing / Content",
                "g": "Helping / Mentoring",
                "h": "Experiments / Science",
                "i": "Coding, apps",
                "j": "Marketing / Selling",
                "k": "Other"
            },
            "category": "enjoyed_activities_multiple"
        },
        "q8": {
            "text": "If you had a YouTube channel, what would it be about?",
            "options": {
                "a": "Science or Tech",
                "b": "Social or Political Awareness",
                "c": "Business / Finance",
                "d": "Art / Creativity",
                "e": "Debates / Motivation",
                "f": "Gaming / Fun"
            },
            "category": "youtube_channel_topic"
        },
        "q9": {
            "text": "What kind of work environment do you see yourself in?",
            "options": {
                "a": "Corporate",
                "b": "Research / Science",
                "c": "Court / Legal",
                "d": "Creative / Media",
                "e": "Education",
                "f": "Government",
                "g": "Healthcare",
                "h": "NGO / Social",
                "i": "Other"
            },
            "category": "preferred_work_environment"
        },
        "q10": {
            "text": "Which optional subject did you choose in Class 12?",
            "options": {
                "a": "CS / IP",
                "b": "Economics",
                "c": "Psychology / History / Political Science",
                "d": "Entrepreneurship",
                "e": "Fine Arts",
                "f": "Physical Ed.",
                "g": "Other"
            },
            "category": "optional_subject_class_12"
        },
        "q11": {
            "text": "Why did you choose that optional subject?",
            "options": {
                "a": "Genuine interest",
                "b": "School provided only these options.",
                "c": "Career prospects",
                "d": "Strong earlier performance",
                "e": "Suggested by peer",
                "f": "Other"
            },
            "category": "reason_for_optional_subject"
        },
        "q12": {
            "text": "Now that you’ve been studying your optional subject, how do you feel about your choice?",
            "options": {
                "a": "Yes — I genuinely enjoy it and it adds to my overall interest or career path",
                "b": "It's okay — not bad, but not something I feel strongly about",
                "c": "No — I don’t find it useful or enjoyable",
                "d": "I chose it for convenience, but I now feel unsure about its value",
                "e": "I’m still figuring it out"
            },
            "category": "optional_subject_experience"
        },
        "q13": {
            "text": "Do you feel that your current stream and subject choices are helping you move in the direction you truly want to go in life?",
            "options": {
                "a": "Yes — I feel happy, focused, and clear that I’m on the right path",
                "b": "Somewhat — I’m not totally sure, but it’s not bad either",
                "c": "No — I feel off-track or unsure"
            },
            "category": "stream_alignment"
        },
        "q13_1": {
            "text": "If your answer is 'No', what do you think is the reason? (Choose any that apply)",
            "options": {
                "a": "I chose based on what my parents wanted",
                "b": "I followed my friends or peer trend",
                "c": "I still haven’t figured out what I really want",
                "d": "The subjects I chose aren’t interesting to me anymore",
                "e": "I feel lost, unmotivated, or confused",
                "f": "Other"
            },
            "category": "reason_for_misalignment_multiple"
        },
        "q14": {
            "text": "Which of the following statements feels most like 'you'?",
            "options": {
                "a": "I like to lead, take charge, and bring change. I want to create a big impact.",
                "b": "I’m focused and calm. I prefer working deeply on things I care about.",
                "c": "Helping, guiding, or supporting people gives me joy. I value connection.",
                "d": "I enjoy building or creating useful things — projects, ideas, or systems.",
                "e": "I’m a balanced person — calm yet a leader, focused yet a team player."
            },
            "category": "personality_style"
        }
    },
    "graduate": {
        "q1": {
            "text": "What degree are you pursuing or have completed?",
            "category": "graduate_degree",
            "options": {
                "a": "B.A. (Humanities, Social Sciences)",
                "b": "B.Com",
                "c": "BBA",
                "d": "B.Sc. (Non-CS)",
                "e": "Other (please specify)"
            }
        },
        "q2": {
            "text": "How do you feel about working in the IT or digital world?",
            "category": "it_digital_world_feeling",
            "options": {
                "a": "I’m very interested and open to learning",
                "b": "Curious but unsure if I’m the right fit",
                "c": "I find it overwhelming",
                "d": "I’ve never really considered it"
            }
        },
        "q3": {
            "text": "What type of tech role would you be most comfortable with?",
            "category": "tech_role_comfort",
            "options": {
                "a": "Minimal or no coding (I prefer using tools, not building them)",
                "b": "A mix — I’m okay with a bit of coding + logic-based work",
                "c": "I’m open to hardcore coding roles if I get proper training",
                "d": "Not sure yet — I need help figuring it out"
            }
        },
        "q4": {
            "text": "Which of these activities feel most natural and enjoyable to you? (Select up to 3)",
            "category": "natural_enjoyable_activities_multiple",
            "options": {
                "a": "Talking to people, explaining things clearly",
                "b": "Preparing documents, organizing information",
                "c": "Making plans, managing small projects",
                "d": "Handling clients and their queries",
                "e": "Working with data, charts, or reports",
                "f": "Understanding markets and customer needs",
                "g": "Designing things (social posts, visuals, presentations)",
                "h": "Exploring apps, websites, or digital tools",
                "i": "Creating content (writing, scripting, etc.)"
            }
        },
        "q5": {
            "text": "Which of these best describes your comfort with technical tools or software?",
            "category": "tech_tools_comfort",
            "options": {
                "a": "I’ve used MS Excel, Google Docs, and a few online tools",
                "b": "I enjoy exploring new apps, dashboards, or platforms",
                "c": "I’ve tried basic coding or logic-based tools (like Scratch, HTML, Excel formulas)",
                "d": "I’ve never tried but I’m eager to learn",
                "e": "I avoid tech stuff as much as I can"
            }
        },
        "q6": {
            "text": "If you had to join a company tomorrow, which kind of role would you feel confident starting in?",
            "category": "confident_starting_role",
            "options": {
                "a": "Support/Client Interaction (solving queries, managing relationships)",
                "b": "Business/Data Analysis (Excel, dashboards, trends)",
                "c": "Marketing or Sales roles (outreach, content, social media)",
                "d": "Content/Documentation/Operations",
                "e": "Project/Team coordination roles",
                "f": "Tech Assistant/Tester (QA, basic testing, feedback)",
                "g": "Not sure yet — that’s why I’m here!"
            }
        },
        "q7": {
            "text": "What's your biggest goal right now?",
            "category": "current_biggest_goal",
            "options": {
                "a": "Get a stable job quickly",
                "b": "Build a long-term career in a growing field",
                "c": "Learn skills that are in demand",
                "d": "Earn well and be independent",
                "e": "Shift from a stuck or unclear career path"
            }
        },
        "q8": {
            "text": "How do you prefer learning new things?",
            "category": "learning_preference",
            "options": {
                "a": "Watching short videos and tutorials",
                "b": "Doing practical, hands-on exercises",
                "c": "Reading and following step-by-step guides",
                "d": "Discussing with mentors or peers",
                "e": "Haven’t figured out my learning style yet"
            }
        },
        "q9": {
            "text": "What kind of work environment do you see yourself thriving in?",
            "category": "work_environment_preference",
            "options": {
                "a": "Flexible and creative",
                "b": "Structured with clear roles",
                "c": "Interactive and people-facing",
                "d": "Calm, task-focused and less social",
                "e": "Remote work with digital tools",
                "f": "Corporate office with team setup"
            }
        },
        "q10": {
            "text": "Lastly, if you had no limitations — what kind of work would you love to do every day?",
            "category": "ideal_work_no_limitations",
            "options": {}
        }
    }
}
