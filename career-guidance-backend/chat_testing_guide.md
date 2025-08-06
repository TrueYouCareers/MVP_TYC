# Chat Functionality Testing Guide

## Prerequisites

1. Ensure you have completed a questionnaire for a user (profile exists in MongoDB)
2. Server is running: `uvicorn app.main:app --reload`
3. Environment variables are set (GROQ_API_KEY, MongoDB connection)

## Testing the Chat System

### 1. Start a New Chat Session

**Request:**

```bash
curl -X POST "http://localhost:8000/guidance/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_12345",
    "question": "Hi! Can you tell me more about my recommended career paths?"
  }'
```

**Actual Response Structure:**

```json
{
  "session_id": "65f8a1b2c3d4e5f6789012ab",
  "response": "Hello! Based on your profile analysis, I can see you have strong analytical and investigative skills. Your primary orientation suggests you'd excel in data-driven fields. Here are some career paths that align well with your interests...",
  "source_documents": [
    {
      "content": "Data science involves extracting insights from large datasets using statistical methods...",
      "metadata": {
        "source": "knowledge_base/careers/data_science.txt"
      }
    },
    {
      "content": "Software engineering requires strong problem-solving skills and logical thinking...",
      "metadata": {
        "source": "knowledge_base/careers/software_engineering.txt"
      }
    }
  ]
}
```

### 2. Continue the Chat Session

**Request:**

```bash
curl -X POST "http://localhost:8000/guidance/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_12345",
    "session_id": "65f8a1b2c3d4e5f6789012ab",
    "question": "What skills should I focus on developing for data science?"
  }'
```

### 3. Get Chat History

**Request:**

```bash
curl -X GET "http://localhost:8000/guidance/chat/history/user_12345/65f8a1b2c3d4e5f6789012ab"
```

**Response Structure:**

```json
{
  "id": "65f8a1b2c3d4e5f6789012ab",
  "user_id": "user_12345",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:35:00Z",
  "messages": [
    {
      "role": "user",
      "content": "Hi! Can you tell me more about my recommended career paths?",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    {
      "role": "assistant",
      "content": "Hello! Based on your profile analysis...",
      "timestamp": "2024-01-15T10:30:05Z"
    }
  ],
  "summary": null
}
```

### 4. Get All User Sessions

**Request:**

```bash
curl -X GET "http://localhost:8000/guidance/chat/sessions/user_12345"
```

## Frontend Integration Example

```javascript
// Updated Chat hook for React that matches actual API response
const useChat = (userId) => {
  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async (question, sessionId = null) => {
    setLoading(true);
    try {
      const response = await fetch("/api/guidance/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          session_id: sessionId,
          question: question, // Note: 'question', not 'message'
        }),
      });

      const result = await response.json();

      // result structure: { session_id, response, source_documents }

      // Add user message to local state
      const userMessage = {
        role: "user",
        content: question,
        timestamp: new Date().toISOString(),
      };

      // Add AI response to local state
      const aiMessage = {
        role: "assistant",
        content: result.response,
        timestamp: new Date().toISOString(),
        source_documents: result.source_documents || [],
      };

      setMessages((prev) => [...prev, userMessage, aiMessage]);

      return {
        session_id: result.session_id,
        message: result.response,
        source_documents: result.source_documents,
      };
    } catch (error) {
      console.error("Chat error:", error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const loadChatHistory = async (sessionId) => {
    try {
      const response = await fetch(
        `/api/guidance/chat/history/${userId}/${sessionId}`
      );
      const session = await response.json();

      // session.messages is an array of { role, content, timestamp }
      setMessages(session.messages || []);
      setCurrentSession(session);

      return session;
    } catch (error) {
      console.error("Error loading chat history:", error);
    }
  };

  const loadSessions = async () => {
    try {
      const response = await fetch(`/api/guidance/chat/sessions/${userId}`);
      const userSessions = await response.json();
      setSessions(userSessions || []);
      return userSessions;
    } catch (error) {
      console.error("Error loading sessions:", error);
      return [];
    }
  };

  return {
    sendMessage,
    loadChatHistory,
    loadSessions,
    sessions,
    currentSession,
    messages,
    loading,
  };
};

// Example ChatArea component usage
const ChatArea = ({ userId }) => {
  const { sendMessage, messages, loading } = useChat(userId);
  const [input, setInput] = useState("");
  const [currentSessionId, setCurrentSessionId] = useState(null);

  const handleSend = async () => {
    if (!input.trim()) return;

    try {
      const result = await sendMessage(input, currentSessionId);
      setCurrentSessionId(result.session_id);
      setInput("");
    } catch (error) {
      console.error("Failed to send message:", error);
    }
  };

  return (
    <div className="chat-area">
      <div className="messages">
        {messages &&
          messages.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              <div className="content">{message.content}</div>
              {message.source_documents &&
                message.source_documents.length > 0 && (
                  <div className="sources">
                    <small>
                      Sources: {message.source_documents.length} documents
                    </small>
                  </div>
                )}
            </div>
          ))}
      </div>

      <div className="input-area">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask your career question..."
          disabled={loading}
        />
        <button onClick={handleSend} disabled={loading || !input.trim()}>
          {loading ? "Sending..." : "Send"}
        </button>
      </div>
    </div>
  );
};
```

## Common Issues & Solutions

### Issue 1: "Cannot read properties of undefined (reading 'map')"

**Root Cause:** Frontend expecting `messages` array but API returns different structure

**Solution:**

1. For chat endpoint: Use `result.response` for the AI message content
2. For chat history: Use `session.messages.map()` for message list
3. Always check if arrays exist before mapping: `messages && messages.map()`

### Issue 2: "User profile not found"

**Solution:** Ensure the user has completed the questionnaire first via `/profile/questionnaire`

### Issue 3: "No retriever available"

**Solution:** Check that the knowledge_base directory exists and contains .txt files

### Issue 4: Empty or generic responses

**Solution:** Verify that the user's LLM profile contains meaningful data and questions_data was provided during questionnaire submission

## Key Points for Frontend Integration

1. **Chat Endpoint Response**: `{ session_id, response, source_documents }`
2. **History Endpoint Response**: `{ id, user_id, messages: [...], created_at, updated_at, summary }`
3. **Message Structure**: `{ role: "user"|"assistant", content: string, timestamp: string }`
4. **Always handle undefined/null arrays before calling `.map()`**
5. **Use `question` field in request body, not `message`**
