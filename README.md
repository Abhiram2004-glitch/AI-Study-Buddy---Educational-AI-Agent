AI Study Buddy - Educational AI Agent
An intelligent study companion that processes your documents and provides personalized learning experiences through AI-powered interactions. This educational agent uses Mistral AI to transform static study materials into dynamic, interactive learning sessions.
What This AI Agent Does
This system functions as an autonomous educational assistant that:

Processes Documents: Automatically extracts and analyzes content from uploaded PDFs
Generates Learning Materials: Creates summaries, quizzes, and coding challenges tailored to your content
Provides Conversational Tutoring: Offers guided study sessions with adaptive feedback
Answers Questions: Uses RAG (Retrieval-Augmented Generation) to respond to queries about your documents
Tracks Progress: Monitors quiz performance and provides personalized recommendations

Core AI Agent Features
🤖 Conversational Study Agent
The AI_Agent.py module implements a conversational AI tutor that:

Maintains conversation state throughout study sessions
Uses function calling to execute educational tools
Provides adaptive feedback based on quiz performance
Guides students through structured learning workflows

📊 Intelligent Content Analysis

Automatically assesses document difficulty levels
Estimates reading time for materials
Identifies key skills needed for comprehension
Generates structured summaries with actionable insights

🎯 Adaptive Assessment Generation

Creates difficulty-appropriate quizzes (Easy/Medium/Hard)
Generates programming challenges based on document content
Provides detailed explanations for learning reinforcement
Offers retake recommendations based on performance

🔍 Smart Document Retrieval

Implements FAISS-based semantic search
Provides contextually relevant answers to user questions
Maintains document context across conversations
Supports multi-document knowledge synthesis

Technical Architecture
AI Agent Components
AI Study Buddy Agent/
├── AI_Agent.py                 # Core conversational agent with tool calling
├── Home.py                     # Document processing and initial analysis
├── pages/
│   ├── Ask_Your_Document_Anything.py  # RAG-powered Q&A system
│   ├── Quiz_Generator.py              # Adaptive quiz generation
│   └── Coding_Questions.py            # Programming challenge creator
├── utils/
│   ├── mistral_utils.py        # AI model integration and prompting
│   └── st_utils.py            # UI components and data processing
└── requirements.txt
AI Models Used

Primary Model: pixtral-12b-2409 (Mistral AI)
Embeddings: mistral-embed for semantic search
Response Format: Structured JSON outputs for reliable parsing

Agent Capabilities
The AI agent implements several key behaviors:

Tool Selection: Automatically chooses appropriate functions based on user intent
Context Retention: Maintains conversation state across interactions
Performance Monitoring: Tracks and responds to student progress
Adaptive Responses: Adjusts difficulty and recommendations based on user performance

Installation and Setup
Prerequisites

Python 3.8+
Mistral AI API key
Streamlit for the web interface

Quick Setup
bash# Clone the repository
git clone <your-repo-url>
cd ai-study-buddy

# Install dependencies
pip install -r requirements.txt

# Configure API key
mkdir .streamlit
echo 'MISTRAL_API_KEY = "your-api-key-here"' > .streamlit/secrets.toml

# Run the agent
streamlit run Home.py
Usage Workflow

Document Upload: Upload PDF study materials
Initial Analysis: AI agent processes and summarizes content
Interactive Learning: Choose from guided study options:

Conversational tutoring sessions
Custom quiz generation
Coding challenge creation
Document Q&A


Progress Tracking: Agent monitors performance and provides feedback
Adaptive Recommendations: Receive personalized study suggestions

AI Agent Features in Detail
Conversational Flow Management
The agent maintains conversation state and guides users through educational workflows:

Initial document summarization
Quiz offer and generation
Performance evaluation and feedback
Retake recommendations for scores below 75%

Intelligent Tool Calling
The agent automatically selects and executes appropriate tools:

summarize_document: Generates comprehensive document summaries
create_quiz: Creates customized quizzes with specified parameters
evaluate_quiz_score: Provides performance feedback and recommendations

Adaptive Learning Pathways
Based on user performance, the agent:

Suggests content review for low quiz scores
Offers advanced challenges for high performers
Provides targeted explanations for missed concepts
Recommends optimal study sequences

Dependencies
txtllama-index-llms-mistralai
PyPDF2
streamlit
mistralai
numpy
faiss-cpu
API Configuration
The agent requires a Mistral AI API key configured in .streamlit/secrets.toml:
tomlMISTRAL_API_KEY = "your-mistral-api-key"
Contributing
This AI agent can be extended with additional capabilities:

New question types and assessment formats
Integration with additional AI models
Enhanced conversation flows
Multi-language support
Advanced analytics and progress tracking

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments
Built using Mistral AI's powerful language models and the Streamlit framework for creating interactive AI applications.
