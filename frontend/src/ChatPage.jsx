import { useState, useRef, useEffect } from 'react';

const API_BASE = 'http://localhost:8000';

function DocBadge({ doc, index }) {
  return (
    <div className="doc-badge">
      <span className="doc-rank">#{index + 1}</span>
      <div className="doc-info">
        <span className="doc-name">{doc.name}</span>
        <span className="doc-type">{doc.type === 'algorithms' ? '算法' : '案例'}</span>
        <span className="doc-similarity">{(doc.similarity * 100).toFixed(1)}%</span>
      </div>
      <p className="doc-summary">{doc.summary}</p>
    </div>
  );
}

function Message({ msg }) {
  return (
    <div className={`message ${msg.role === 'user' ? 'message-user' : 'message-bot'}`}>
      <div className="message-role">{msg.role === 'user' ? 'You' : 'Solver Agent'}</div>
      <div className="message-content">
        {msg.answer && <div className="answer-text">{msg.answer}</div>}
        {msg.error && <div className="answer-error">{msg.error}</div>}
        {msg.results && msg.results.length > 0 && (
          <details className="results-panel">
            <summary>Retrieved {msg.results.length} documents</summary>
            <div className="doc-list">
              {msg.results.map((doc, i) => (
                <DocBadge key={doc.path} doc={doc} index={i} />
              ))}
            </div>
          </details>
        )}
        {msg.elapsedMs != null && (
          <div className="elapsed">{msg.elapsedMs < 1000 ? `${msg.elapsedMs.toFixed(0)}ms` : `${(msg.elapsedMs / 1000).toFixed(1)}s`}</div>
        )}
      </div>
    </div>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState(5);
  const [docType, setDocType] = useState('');
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    const query = input.trim();
    if (!query || loading) return;

    const userMsg = { role: 'user', content: query };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const body = { query, top_k: topK };
      if (docType) body.type = docType;

      const res = await fetch(`${API_BASE}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: query,
          answer: data.answer,
          results: data.results,
          elapsedMs: data.elapsed_ms,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: query, error: err.message },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-page">
      <div className="chat-controls">
        <div className="control-item">
          <label>Top-K</label>
          <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
            {[1, 3, 5, 10, 20].map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
        </div>
        <div className="control-item">
          <label>类型</label>
          <select value={docType} onChange={(e) => setDocType(e.target.value)}>
            <option value="">全部</option>
            <option value="algorithms">算法</option>
            <option value="projects">案例</option>
          </select>
        </div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <h2>Solver Agent</h2>
            <p>Describe your optimization problem, and I'll recommend algorithms and cases from the knowledge base.</p>
            <div className="example-queries">
              <span>Try:</span>
              {[
                '如何解决车辆路径规划问题？',
                '多目标生产调度推荐什么算法？',
                '机器人任务分配用哪个案例参考？',
              ].map((q) => (
                <button key={q} onClick={() => setInput(q)}>{q}</button>
              ))}
            </div>
          </div>
        )}
        {messages.map((msg, i) => (
          <Message key={i} msg={msg} />
        ))}
        {loading && (
          <div className="message message-bot">
            <div className="message-role">Solver Agent</div>
            <div className="loading-dots">
              <span /><span /><span />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-area">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe your optimization problem..."
          rows={3}
          disabled={loading}
        />
        <button onClick={handleSend} disabled={loading || !input.trim()}>
          {loading ? 'Thinking...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
