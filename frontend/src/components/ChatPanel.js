import {useState} from 'react'

function ChatPanel() {
    const [message, setMessage] = useState('')
    const [response, setResponse] = useState('');

    const sendMessage = async () => {
        const res = await fetch('http://localhost:8000/ask_rag_node', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({question: message}),
        });
        const data = await res.json();
        setResponse(data.answer);
    };

    return (
        <div className="chat-panel">
            <textarea value={message} onChange={(e) => setMessage(e.target.value)}></textarea>
            <button onClick={sendMessage}>Send</button>
            <div><strong>Odpowiedź:</strong> {response}</div>
        </div>
    )
}

export default ChatPanel