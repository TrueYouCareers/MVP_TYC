from pymongo.database import Database
from app.schemas.report import ReportData
from app.schemas.questionnaire import QuestionnaireInDB
from app.llm.rag_system import RAGSystem
from app.llm.prompts import Prompts
from app.data.user_question import QUESTIONNAIRE_DATA
from datetime import datetime
import json
from typing import Dict, Any, Optional, List
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
# This import is correct. Ensure 'reportlab' is installed.
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from playwright.async_api import async_playwright
from app.schemas.question import QuestionCreate, QuestionSchema


class ReportService:
    def __init__(self, mongo_db: Database, rag_system: RAGSystem):
        self.mongo_db = mongo_db
        self.rag_system = rag_system

    async def generate_report(self, user_id: str) -> ReportData:
        profile_doc = self.mongo_db["user_questionnaires"].find_one(
            {"user_id": user_id}, sort=[("timestamp", -1)]
        )
        if not profile_doc:
            raise ValueError(f"User profile not found for user_id: {user_id}")

        try:
            llm_profile_data = profile_doc.get("llm_profile")
            education_level = profile_doc.get("education_level")
            raw_responses = profile_doc.get("raw_responses")
        except Exception as e:
            raise ValueError(f"Failed to extract profile fields: {e}")

        if not llm_profile_data or not isinstance(llm_profile_data, dict):
            raise ValueError(
                f"LLM generated profile data is missing or invalid for user_id: {user_id}"
            )

        # Get latest chat summary
        latest_session_doc = self.mongo_db["chat_sessions"].find_one(
            {"user_id": user_id}, sort=[("updated_at", -1)]
        )
        chat_summary_text = "No recent chat session found."
        if latest_session_doc:
            chat_summary_text = latest_session_doc.get("summary")
            if not chat_summary_text and latest_session_doc.get("messages"):
                chat_summary_text = f"Chat session with {len(latest_session_doc['messages'])} messages. Detailed summary pending."
            elif not chat_summary_text:
                chat_summary_text = "Chat session found but not yet summarized."

        report_prompt_template = Prompts.get_report_generation_prompt()

        try:
            user_profile_json_str = json.dumps(llm_profile_data)
        except TypeError:
            raise ValueError("Could not serialize LLM profile data to JSON for report generation.")

        formatted_prompt = report_prompt_template.format(
            user_profile_json=user_profile_json_str,
            chat_summary_text=chat_summary_text
        )

        try:
            llm_response = self.rag_system.llm.invoke(formatted_prompt)
            llm_report_content_str = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            generated_report_dict = json.loads(llm_report_content_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM did not return valid JSON for the report. Error: {e}")
        except Exception as e:
            raise RuntimeError(f"Error invoking LLM for report generation: {e}")

        return ReportData(
            user_id=user_id,
            profile_snapshot=llm_profile_data,
            report_title=generated_report_dict.get("report_title", "Personalized Career Guidance Report"),
            introduction=generated_report_dict.get("introduction"),
            profile_analysis_summary=generated_report_dict.get("profile_analysis", {}).get("summary"),
            strengths_and_interests_detail=generated_report_dict.get("profile_analysis", {}).get("strengths_and_interests"),
            personality_insights_detail=generated_report_dict.get("profile_analysis", {}).get("personality_insights"),
            career_path_explorations=generated_report_dict.get("career_path_exploration", []),
            chat_session_recap=generated_report_dict.get("chat_session_recap"),
            actionable_recommendations=generated_report_dict.get("actionable_recommendations", []),
            concluding_remarks=generated_report_dict.get("concluding_remarks"),
            generated_at=datetime.utcnow()
        )


    async def generate_llm_filled_template_report_string(self, user_id: str) -> str:
        profile_doc = self.mongo_db["user_questionnaires"].find_one(
            {"user_id": user_id}, sort=[("timestamp", -1)]
        )
        if not profile_doc:
            raise ValueError(f"User profile not found for user_id: {user_id}")

        try:
            llm_profile_data = profile_doc.get("llm_profile")
            education_level = profile_doc.get("education_level", "graduate").lower()
            raw_responses = profile_doc.get("raw_responses")
        except Exception as e:
            raise ValueError(f"Failed to extract profile fields: {e}")

        if not llm_profile_data or not isinstance(llm_profile_data, dict):
            raise ValueError(f"LLM generated profile data is missing or invalid for user_id: {user_id}")

        # Fetch chat history
        chat_sessions = self.mongo_db["chat_sessions"].find({"user_id": user_id}).sort("updated_at", -1)
        formatted_chat_history = "No chat history found."

        all_messages_str_list = []
        for session in chat_sessions:
            messages = session.get("messages", [])
            for msg in messages:
                sender = msg.get("sender", "Unknown")
                content = msg.get("content", "")
                all_messages_str_list.append(f"{sender}: {content}")

        if all_messages_str_list:
            formatted_chat_history = "\n".join(all_messages_str_list)
        
        # updated the question fetch from database
        temp = self.mongo_db["questions"].find({"education_level": education_level}).sort("order", 1)
        docs = temp.to_list(length=None)

        questionnaire_set = [QuestionSchema(**doc) for doc in docs]



        # Convert the list to a dictionary using question name as key for quick lookup
        question_map = {q.name: q for q in questionnaire_set}

        answered_questions_summary = []

        if raw_responses and isinstance(raw_responses, dict):
            for q_name, selected_option in raw_responses.items():
                question_data = question_map.get(q_name)
                if question_data:
                    question_text = question_data.prompt
                    # Handle 'text' or 'multi' types which might not have options
                    if question_data.type in ['text', 'multi']:
                        selected_answer_text = selected_option
                    else:
                        # If type is 'single' and option is from options list
                        selected_answer_text = selected_option if selected_option in question_data.options else "Invalid option"
                    answered_questions_summary.append(
                        f"{question_data.title} {question_text}\nAnswer: {selected_answer_text}\n"
                    )
                else:
                    answered_questions_summary.append(
                        f"{q_name}. [Unknown Question]\nAnswer: {selected_option}\n"
                    )

        formatted_questionnaire_section = "\n".join(answered_questions_summary) or "No questionnaire responses available."


        # print(formatted_questionnaire_section)
        # Select template based on education level
        if education_level in ("9th", "10th"):
            template_to_fill = """
            📘 Hello {{Student Name if available, otherwise 'Student'}},

We’re excited to share your personalized guidance report based on your inputs. Here's what we discovered about you!

🧭 1. Your Natural Inclination & Ideal Careers  
From what you shared (Q1, Q2, Q4, Q6), you feel confident in {{Subject Confidence — Q1}} and are naturally drawn to {{Stream Interest — Q2}}.  
Your dream profession is {{Dream Career — Q4}}, and you imagine yourself working in an environment that is {{Work Style — Q6}}.

This points toward career areas where your interests and mindset can thrive — such as:
✅ {{Career Suggestion 1}}, {{Career Suggestion 2}}, and {{Career Suggestion 3}}.

💡 2. Your Strengths, Skills & Personality  
Looking at your responses to Q5, Q7, Q10, and Q11, we noticed that you enjoy:
- 🛠️ Hobbies like {{Hobbies — Q5}}  
- 🕒 Activities you do in your free time: {{Free Time Activities — Q7}}  
- 🧵 Topics you’d create content about: {{Content Creation Topic — Q10}}  
- 🌀 Things that make you lose track of time: {{Flow Activities — Q11}}

These tell us you’re someone who is {{summary personality type — e.g., creative thinker, logical builder, etc.}}, with strengths in {{List of skills based on responses}}.

🎨 3. Do Your Interests Support Your Career Path?  
We mapped your stream interest (Q2), hobbies (Q5), and flow activities (Q11).  
✅ If aligned: That’s amazing! You’re on a natural path that fits who you are.  
🔁 If not fully aligned: Don’t worry — many students discover hybrid paths, like combining {{Interest A}} and {{Interest B}}, into careers like {{Example Hybrid Career}}.

🏫 4. Your Engagement in School Life  
Your response to Q9 indicates that {{Engagement Summary — e.g., you’re exploring, actively involved, or unsure}}.

You might enjoy trying out:
🎯 School clubs like {{Suggested Clubs}},  
📢 Competitions in {{Suggested Competitions}},  
🎨 Creative challenges in {{Suggested Creative Opportunities}}.

Even if you're unsure right now, trying something new this term could help you discover hidden interests.

🚦 5. Are You Feeling Confident About Your Path?  
From Q14, you shared that you feel {{Confidence Level}}.  
{{If "No"}}: That’s okay — a lot of students feel the same. Your reasons — like {{Selected reasons: e.g., “pressure from parents” or “lack of clarity”}} — are valid.

Here’s a gentle reminder:
✨ “It’s okay to feel unsure — many students do. The important thing is to stay curious and explore step by step.”

🎁 6. Your Next Steps  
To grow and explore your future path, you could:
- 🔍 Explore more via your school clubs or science fair
- 💬 Talk to a teacher or mentor about what excites you
- 📅 Book a one-on-one session with TrueYouCareers for deeper clarity
- 🎤 Try participating in {{Activity Suggestion}} to test your interest

---

You’re not expected to have all the answers today — but every answer you gave helps shape your direction. Keep exploring, stay curious, and believe in yourself!

Warm wishes,  
✨ Team TrueYouCareers

"""

        elif education_level in ("11th", "12th"):
            template_to_fill = """
            📘 Hello {{Student Name if available, otherwise 'Student'}},

Here’s your personalized career reflection report, built from your responses and interests. Let’s take a look at what’s emerging for you!

🧭 Your Natural Inclination  
Based on your responses, you show a strong natural inclination toward:
👉 {{Suggested Stream or Domain}}

You're currently pursuing {{Chosen Stream}}, and if given full choice, you'd love to study {{Q4 Subjects}} — a great signal of where your energy truly lies.

🎯 Your Career Direction  
You see yourself becoming a {{Dream Career}}, which is a bold and inspiring vision.

This aligns well with your interest in {{Q6 Subjects}}, your active engagement in {{Q7 Activities}}, and your curiosity to create or speak about topics like {{Content Topic — Q10}}.

These choices indicate you’re already thinking in a direction that’s meaningful to you.

💡 Your Qualities That Will Help You Succeed  
You naturally show signs of being:
- 💡 {{Trait 1 — e.g., Logical Thinker, Empathetic Leader, Expressive Communicator}}
- ⚙️ {{Trait 2 — e.g., Curious Learner, Tech Explorer, Deep Worker}}

These personal strengths will help you thrive in your chosen path.

Your personality — “{{Q14 Personality Statement}}” — adds even more power to your journey. It's something unique you bring into any environment.

⚙️ Your Optional Subject — Make the Most of It  
You've selected {{Optional Subject}} as your additional subject.  
Here's what we recommend:
- If you're excited by it: Lean in and go deep — build skills, try projects, explore real-world applications.
- If you're unsure: Give it your best shot. Even one valuable takeaway could help in your career.
- If it’s tech-related and you're curious: Dive into coding, digital tools, or hands-on practice. It builds your digital fluency — a powerful skill in any field today.

🚧 Any Roadblocks?  
You mentioned feeling {{Q13 Alignment}} about your path.

{% if not_aligned %}
You also shared challenges like: {{Q13 Blockers}}  
→ Don’t keep those doubts inside. Speak to a mentor, counselor, or book a session with us. Many students face these — but the ones who move forward are the ones who reflect and reach out.
{% endif %}

📝 Suggestions & Exploration Ideas  
Based on your profile, we recommend:
- 🎓 Career Options to Explore: {{Career Clusters}}
- 📚 Entrance Exams to Consider: {{Exam Suggestions}}
- 🎤 Activities to Try: {{Q9 School Activity}} or related clubs/competitions

💭 Keep asking yourself: *“Does this truly excite me?”*

🌟 Final Thought  
You’re not expected to have it all figured out — you’re expected to begin.  
With awareness, curiosity, and a bit of courage — you’re already on your way.

We believe in your journey.  
– Team TrueYouCareers

"""
        else:  # Default to graduate
            template_to_fill = """Template For Graduates

Hello {{Student Name if available, otherwise 'User'}}, here's what we’ve figured out about you!

👩‍🎓 Your Background:
You’ve completed or are pursuing {{Degree}}.
Your mindset shows curiosity toward the {{Identified Interest Area, e.g., digital/IT space}}, and that’s a great first step!

🌿 What You’re Naturally Good At:
Based on your responses, you enjoy or are confident in:
✅ {{selected_natural_activities_1}}, {{selected_natural_activities_2}}, {{selected_natural_activities_3}}
That tells us a lot about your workstyle and strength areas — you’re someone who:

🧠 Works best when doing {{e.g., client interaction / organizing / planning / explaining / analyzing}}
💬 Communicates clearly and understands structure
🔧 Is willing to learn tools and adapt — a skill in huge demand today!

💡 Best-Suited Career Tracks for You:
Based on your natural inclination and tech comfort, here are 2–3 career paths you’ll likely enjoy and succeed in:

🎯 {{Career Path 1}} — e.g., Digital Marketing Executive / Campaign Analyst
📊 {{Career Path 2}} — e.g., Business Analyst / Excel Reporting Associate
💬 {{Career Path 3}} — e.g., Client Relationship Executive / Pre-Sales Associate
These are roles where tech is your friend — not your fear. They don’t require hardcore coding but do need logical thinking, comfort with tools, and the soft skills you already have.

🛠️ Next Step: What to Learn Now
Here’s what you can start learning to become job-ready in these roles:

🧠 Suggested Skills: {{Excel / Google Sheets, Digital Marketing, CRM tools, Business Communication, etc.}}
🎓 Suggested Certifications: {{List optional (if using affiliate links)}}
⏳ Many of these can be learned in just 4–8 weeks, even while continuing your degree/job prep.

🔁 Still thinking about Gov exams or confused about tech?
You can continue your exam prep — but build these modern skills side-by-side.
The job market is shifting fast. Smart graduates prepare a Plan B that pays.

✨ Final Remark:
You’re not behind — you just need direction. And this is it.
With your strengths, you can step into the digital world confidently — not as a coder, but as a communicator, analyst, planner, or team support pro.
"""

#         prompt_for_llm = f"""
#         You are an AI career guidance assistant. Your task is to thoughtfully fill in the following report template using three key sources:
# 1. The student's LLM-generated profile
# 2. Their questionnaire responses
# 3. Their chat history with the assistant

# Replace all placeholders such as `{{...}}` and `[Auto-filled...]` with meaningful, personalized, and well-synthesized content derived from the student's data.

# Please follow these rules strictly:
# - Preserve the formatting and structure of the template exactly as provided.
# - Do not invent information that is not supported by the input data.
# - Ensure that the tone remains friendly, encouraging, and student-focused.
# - Output only the final filled report text — do not include any headers, explanations, or code blocks.
# """
        prompt_for_llm = f"""
        You are an AI career guidance assistant. Your task is to thoughtfully fill in the following report **template** using:

1. The student's LLM-generated profile  
2. Their questionnaire responses  
3. Their chat history  

🎯 Instructions:
- Replace all placeholders like `{{...}}` and `[Auto-filled...]` with meaningful, personalized content.
- Preserve **section structure and tone** from the template.
- Output the **final report strictly in well-structured HTML**, using `<h2>`, `<p>`, `<ul>`, `<li>`, and emojis as needed.
- Use `<style>` tags if needed for inline CSS.
- Do NOT include any explanations, markdown, or code blocks — output pure HTML only.
- Do not include or mention question number in the final report.Ex- Don't mention Q1, Q2 ...

🎨 Additional Formatting Requirements:
- Include a `<header>` at the top of the report with the title: **"TrueYou Careers"**
- Set the background color of the page to `#FEF9C3` using inline CSS (for example, via the `<body>` style)


Student's LLM Profile:
```json
{json.dumps(llm_profile_data, indent=2)}
```
Student's Questionnaire Responses:
```
{formatted_questionnaire_section}
```
Student's Chat History (User and AI conversation):
```
{formatted_chat_history}
```

Report Template:
{template_to_fill}
"""

        try:
            llm_response = self.rag_system.llm.invoke(prompt_for_llm)
            llm_filled_report_str = llm_response.content if hasattr(
                llm_response, 'content') else str(llm_response)
        except Exception as e:
            raise RuntimeError(
                f"Error invoking LLM for template report generation: {e}")

        return llm_filled_report_str

    def create_pdf_report(self, report_string: str) -> bytes:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        for paragraph in report_string.strip().split("\n\n"):
            story.append(Paragraph(paragraph.strip().replace("\n", "<br/>"), styles["Normal"]))
            story.append(Spacer(1, 12))

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes


    async def html_to_pdf_with_playwright(self, html: str) -> bytes:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.set_content(html, wait_until="networkidle")
            pdf_bytes = await page.pdf(format="A4", print_background=True)
            await browser.close()
            return pdf_bytes