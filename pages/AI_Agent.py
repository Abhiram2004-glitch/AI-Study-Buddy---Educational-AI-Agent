import streamlit as st
import PyPDF2
import json
import time
import random
from mistralai import Mistral
from utils.mistral_utils import generate_summary, generate_quiz, generate_coding_questions

st.set_page_config(page_title="🤖 AI Study Tutor", page_icon="🤖")

class ConversationalStudyTutor:
    def __init__(self, api_key):
        self.client = Mistral(api_key=api_key)
        self.model = "pixtral-12b-2409"
        self.documents = []
        self.conversation_state = "initial"  # initial, summarized, quiz_offered, quiz_taken
        self.quiz_score = None
        self.max_retries = 3
        self.base_delay = 1
        
    def set_documents(self, documents):
        self.documents = documents
        
    def wait_with_exponential_backoff(self, attempt):
        delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
        time.sleep(delay)
        return delay
        
    def get_available_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "summarize_document",
                    "description": "Generate a comprehensive summary of a document",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_name": {
                                "type": "string",
                                "description": "Name of the document to summarize"
                            }
                        },
                        "required": ["doc_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_quiz",
                    "description": "Create a quiz with specified number of questions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_name": {
                                "type": "string",
                                "description": "Name of the document"
                            },
                            "num_questions": {
                                "type": "integer",
                                "description": "Number of questions to generate"
                            },
                            "difficulty": {
                                "type": "string",
                                "description": "Difficulty level",
                                "enum": ["Easy", "Medium", "Hard"],
                                "default": "Medium"
                            }
                        },
                        "required": ["doc_name", "num_questions"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "evaluate_quiz_score",
                    "description": "Evaluate quiz performance and provide feedback",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "score_percentage": {
                                "type": "number",
                                "description": "Quiz score as percentage"
                            }
                        },
                        "required": ["score_percentage"]
                    }
                }
            }
        ]
    
    def execute_tool(self, tool_name, parameters):
        if tool_name == "summarize_document":
            doc_name = parameters.get("doc_name")
            doc = next((d for d in self.documents if d["name"] == doc_name), None)
            if doc:
                summary = generate_summary(doc["content"])
                self.conversation_state = "summarized"
                return f"Document Summary Generated:\n{summary}"
            return f"Document '{doc_name}' not found."
        
        elif tool_name == "create_quiz":
            doc_name = parameters.get("doc_name")
            num_questions = parameters.get("num_questions", 5)
            difficulty = parameters.get("difficulty", "Medium")
            doc = next((d for d in self.documents if d["name"] == doc_name), None)
            if doc:
                quiz = generate_quiz(doc["content"], num_questions, difficulty)
                self.conversation_state = "quiz_created"
                return f"Quiz Created with {num_questions} questions:\n{quiz}"
            return f"Document '{doc_name}' not found."
        
        elif tool_name == "evaluate_quiz_score":
            score = parameters.get("score_percentage")
            self.quiz_score = score
            self.conversation_state = "quiz_evaluated"
            if score >= 75:
                return f"Excellent work! You scored {score}%. You have a good understanding of the material."
            else:
                return f"You scored {score}%. I recommend reviewing the material again before retaking the quiz. The passing threshold is 75%."
        
        return f"Unknown tool: {tool_name}"
    
    def make_api_call_with_retry(self, messages, tools):
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
                return response, None
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    if attempt < self.max_retries - 1:
                        delay = self.wait_with_exponential_backoff(attempt)
                        st.warning(f"Rate limit hit. Retrying in {delay:.1f} seconds...")
                        continue
                    else:
                        return None, "Rate limit exceeded. Please try again later."
                else:
                    return None, f"API error: {error_str}"
        
        return None, "Maximum retries reached"
    
    def chat(self, user_message, conversation_history):
        # Enhanced system message for conversational tutoring
        system_message = f"""You are an intelligent AI tutor having a conversation with a student. Your role is to guide them through their learning journey.

Current conversation state: {self.conversation_state}
Available documents: {', '.join([doc['name'] for doc in self.documents])}

Follow this educational workflow:
1. When student asks to study/summarize documents, use the summarize_document tool
2. After summarizing, offer to create a quiz to test their understanding
3. If they want a quiz, ask how many questions they'd like (suggest 5-10)
4. After they take the quiz, they'll tell you their score - use evaluate_quiz_score tool
5. If score < 75%, encourage them to review material and retake quiz
6. If score >= 75%, congratulate them and offer to help with other documents

Be conversational, encouraging, and act like a personal tutor. Don't just execute tools - engage in dialogue."""

        # Add conversation history to messages
        messages = [{"role": "system", "content": system_message}]
        
        # Add previous conversation
        for msg in conversation_history:
            messages.append(msg)
            
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        max_iterations = 3
        iteration = 0
        
        while iteration < max_iterations:
            response, error = self.make_api_call_with_retry(messages, self.get_available_tools())
            
            if error:
                return f"Error: {error}", messages
            
            assistant_message = response.choices[0].message
            
            # Add assistant response to conversation
            messages.append({
                "role": "assistant", 
                "content": assistant_message.content,
                "tool_calls": assistant_message.tool_calls
            })
            
            # Handle tool calls
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    parameters = json.loads(tool_call.function.arguments)
                    tool_result = self.execute_tool(tool_name, parameters)
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "name": tool_name,
                        "content": tool_result,
                        "tool_call_id": tool_call.id
                    })
                
                iteration += 1
                time.sleep(0.5)
                continue
            
            # Return final response
            return assistant_message.content, messages
        
        return "Maximum iterations reached", messages

@st.cache_resource
def get_study_tutor():
    try:
        return ConversationalStudyTutor(st.secrets["MISTRAL_API_KEY"])
    except KeyError:
        st.error("Please set your MISTRAL_API_KEY in Streamlit secrets.")
        return None

# Main interface
st.title("🤖 AI Study Tutor")
st.info("Your personal AI tutor that guides you through summarization, quizzes, and assessment")

tutor = get_study_tutor()
if not tutor:
    st.stop()

# Initialize session state
if "documents" not in st.session_state:
    st.session_state.documents = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "awaiting_quiz_score" not in st.session_state:
    st.session_state.awaiting_quiz_score = False

# File upload
uploaded_files = st.file_uploader("Upload PDFs for Study Session", 
                                 type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("📤 Process Files"):
        with st.spinner("Processing PDFs..."):
            documents = []
            for uploaded_file in uploaded_files:
                try:
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    documents.append({"name": uploaded_file.name, "content": text})
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            if documents:
                st.session_state.documents = documents
                tutor.set_documents(documents)
                st.success(f"Processed {len(documents)} documents! Start your study session below.")

# Update tutor with current documents
if st.session_state.documents:
    tutor.set_documents(st.session_state.documents)
    
    # Show available documents
    with st.expander("📂 Available Documents"):
        for doc in st.session_state.documents:
            st.write(f"• {doc['name']}")
    
    # Chat interface
    st.markdown("### 💬 Study Session")
    
    # Display conversation history
    for msg in st.session_state.conversation_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Handle quiz display if quiz was created
    if st.session_state.quiz_data and not st.session_state.awaiting_quiz_score:
        st.markdown("### 📝 Take Your Quiz")
        
        try:
            quiz_questions = json.loads(st.session_state.quiz_data)
            
            with st.form("quiz_form"):
                st.write("Answer all questions and submit to get your score:")
                
                answers = {}
                for i, q in enumerate(quiz_questions):
                    st.subheader(f"Question {i+1}: {q['Question']}")
                    answers[i] = st.radio(
                        f"Choose answer for Q{i+1}:",
                        q['Options'],
                        key=f"q_{i}",
                        index=None
                    )
                
                submitted = st.form_submit_button("Submit Quiz")
                
                if submitted:
                    # Calculate score
                    correct = 0
                    total = len(quiz_questions)
                    
                    for i, q in enumerate(quiz_questions):
                        if answers[i] == q['Answer']:
                            correct += 1
                    
                    score_percentage = (correct / total) * 100
                    
                    # Display results
                    st.markdown(f"### Results: {correct}/{total} ({score_percentage:.1f}%)")
                    
                    for i, q in enumerate(quiz_questions):
                        if answers[i] == q['Answer']:
                            st.success(f"Q{i+1}: Correct!")
                        else:
                            st.error(f"Q{i+1}: Wrong. Correct answer: {q['Answer']}")
                    
                    # Add score to conversation
                    st.session_state.conversation_history.append({
                        "role": "user", 
                        "content": f"I completed the quiz and scored {score_percentage:.1f}%"
                    })
                    
                    # Clear quiz data and mark as awaiting score processing
                    st.session_state.quiz_data = None
                    st.session_state.awaiting_quiz_score = True
                    st.rerun()
                        
        except json.JSONDecodeError:
            st.error("Error displaying quiz. Please generate a new quiz.")
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to history
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, updated_messages = tutor.chat(prompt, st.session_state.conversation_history)
                st.write(response)
                
                # Check if response contains quiz data
                if "Quiz Created" in response and "{" in response:
                    # Extract quiz JSON from response
                    try:
                        quiz_start = response.find("[")
                        quiz_end = response.rfind("]") + 1
                        if quiz_start != -1 and quiz_end != 0:
                            quiz_json = response[quiz_start:quiz_end]
                            st.session_state.quiz_data = quiz_json
                    except:
                        pass
        
        # Update conversation history with assistant response
        st.session_state.conversation_history.append({"role": "assistant", "content": response})
        
        # Reset quiz score awaiting flag
        if st.session_state.awaiting_quiz_score:
            st.session_state.awaiting_quiz_score = False
            
        st.rerun()

    # Starter prompts
    if not st.session_state.conversation_history:
        st.markdown("**💡 Try these starter prompts:**")
        starter_prompts = [
            "Please summarize my documents",
            "Help me study this material",
            "I want to test my understanding with a quiz"
        ]
        
        for prompt in starter_prompts:
            if st.button(prompt, key=f"starter_{prompt}"):
                st.session_state.conversation_history.append({"role": "user", "content": prompt})
                st.rerun()

else:
    st.warning("Please upload PDF documents to start your study session.")
    
    st.markdown("""
    ### How it works:
    1. **Upload** your study materials (PDFs)
    2. **Ask for summaries** of your documents  
    3. **Take quizzes** to test your understanding
    4. **Get feedback** and recommendations based on your performance
    5. **Repeat** until you achieve mastery (75%+ quiz scores)
    """)