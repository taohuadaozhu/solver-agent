import { useState, useRef, useEffect } from 'react';

const API_RECOMMEND = 'http://localhost:8000';
const API_WORKFLOW = 'http://localhost:8001';

const STEP_LABELS = {
  STEP_1_PROBLEM: 'Problem Understanding',
  STEP_2_DATASET: 'Dataset Selection',
  STEP_3_VARIABLE: 'Decision Variables',
  STEP_4_OBJECTIVE: 'Objective Definition',
  STEP_5_CONSTRAINT: 'Constraints',
  STEP_6_CLASSIFY: 'Problem Classification',
  STEP_7_ALGO: 'Algorithm Recommendation',
};

const STEP_NAMES = Object.keys(STEP_LABELS);

function DocBadge({ doc, index }) {
  return (
    <div className="doc-badge">
      <span className="doc-rank">#{index + 1}</span>
      <div className="doc-info">
        <span className="doc-name">{doc.name}</span>
        <span className="doc-type">{doc.type === 'algorithms' ? '算法' : doc.type === 'datasets' ? '数据' : '案例'}</span>
        <span className="doc-similarity">{(doc.similarity * 100).toFixed(1)}%</span>
      </div>
      <p className="doc-summary">{doc.summary}</p>
    </div>
  );
}

function ChartView({ chart }) {
  if (!chart || !chart.data) return null;
  const id = `chart-${Math.random().toString(36).slice(2)}`;

  useEffect(() => {
    if (window.Plotly) {
      window.Plotly.newPlot(id, chart.data, chart.layout, {
        responsive: true,
        displayModeBar: false,
        margin: { t: 40, r: 20, b: 40, l: 40 },
      });
    }
  }, [id, chart]);

  return <div id={id} className="chart-container" />;
}

function StepProgress({ stepResults, currentStep }) {
  return (
    <div className="step-progress">
      {STEP_NAMES.map((step, i) => {
        const done = stepResults[step] != null;
        const active = step === currentStep;
        return (
          <div key={step} className={`step-item ${done ? 'step-done' : ''} ${active ? 'step-active' : ''}`}>
            <div className="step-indicator">{done ? '✓' : active ? '●' : i + 1}</div>
            <span className="step-label">{STEP_LABELS[step]}</span>
          </div>
        );
      })}
    </div>
  );
}

function WorkflowResult({ state, onExecute }) {
  const { step_results, rag_docs, execution_result, charts } = state;
  const algo = state.recommended_algorithm;
  const dataset = state.dataset_name;

  return (
    <div className="workflow-result">
      {/* Steps accordion */}
      {STEP_NAMES.map((step) => {
        const content = step_results?.[step];
        if (!content) return null;
        return (
          <details key={step} className="step-detail" open={step === 'STEP_7_ALGO'}>
            <summary>
              <strong>{STEP_LABELS[step]}</strong>
              {step === 'STEP_2_DATASET' && dataset && <span className="step-badge">→ {dataset}</span>}
              {step === 'STEP_7_ALGO' && algo && <span className="step-badge">→ {algo}</span>}
            </summary>
            <pre className="step-content">{content}</pre>
          </details>
        );
      })}

      {/* RAG Documents */}
      {rag_docs && rag_docs.length > 0 && (
        <details className="rag-panel" open>
          <summary>Retrieved {rag_docs.length} documents from knowledge base</summary>
          <div className="doc-list">
            {rag_docs.slice(0, 8).map((doc, i) => (
              <DocBadge key={`${doc.name}-${i}`} doc={doc} index={i} />
            ))}
          </div>
        </details>
      )}

      {/* Execute button */}
      {algo && !execution_result && (
        <button className="execute-btn" onClick={() => onExecute(algo, dataset)}>
          Execute {algo} on {dataset || 'selected dataset'}
        </button>
      )}

      {/* Execution result */}
      {execution_result && (
        <div className="execution-result">
          <h3>Execution Result</h3>
          <div className="exec-metrics">
            <div className="metric">
              <span className="metric-value">{execution_result.best_objective}</span>
              <span className="metric-label">Best Objective</span>
            </div>
            <div className="metric">
              <span className="metric-value">{execution_result.runtime_seconds}s</span>
              <span className="metric-label">Runtime</span>
            </div>
            <div className="metric">
              <span className="metric-value">{execution_result.iterations}</span>
              <span className="metric-label">Iterations</span>
            </div>
            <div className="metric">
              <span className={`metric-value quality-${execution_result.solution_quality}`}>
                {execution_result.solution_quality}
              </span>
              <span className="metric-label">Quality</span>
            </div>
          </div>
          <pre className="exec-output">{execution_result.raw_output}</pre>
        </div>
      )}

      {/* Charts */}
      {charts && charts.length > 0 && charts.map((chart, i) => (
        <ChartView key={i} chart={chart} />
      ))}
    </div>
  );
}

function Message({ msg, onExecute }) {
  return (
    <div className={`message ${msg.role === 'user' ? 'message-user' : 'message-bot'}`}>
      <div className="message-role">{msg.role === 'user' ? 'You' : 'Solver Agent'}</div>
      <div className="message-content">
        {msg.role === 'user' && <div className="user-text">{msg.content}</div>}
        {msg.loading && (
          <div className="workflow-loading">
            <StepProgress stepResults={msg.stepResults || {}} currentStep={msg.currentStep} />
            <div className="loading-dots"><span /><span /><span /></div>
          </div>
        )}
        {msg.error && <div className="answer-error">{msg.error}</div>}
        {msg.workflowState && <WorkflowResult state={msg.workflowState} onExecute={onExecute} />}
        {msg.elapsedMs != null && !msg.loading && (
          <div className="elapsed">
            {msg.elapsedMs < 1000 ? `${msg.elapsedMs.toFixed(0)}ms` : `${(msg.elapsedMs / 1000).toFixed(1)}s`}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('workflow'); // 'workflow' | 'quick'
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleExecute = async (algorithm, datasetName) => {
    if (!algorithm) return;
    setLoading(true);

    const loadingMsg = {
      role: 'assistant',
      content: `execute-${algorithm}`,
      loading: true,
      currentStep: 'EXECUTE',
    };
    setMessages((prev) => [...prev, loadingMsg]);

    try {
      const res = await fetch(`${API_WORKFLOW}/workflow/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          algorithm,
          dataset_name: datasetName || '',
          parameters: { max_iterations: 200, population_size: 50 },
        }),
      });

      if (!res.ok) throw new Error(`Execute failed: HTTP ${res.status}`);
      const data = await res.json();

      setMessages((prev) => {
        const updated = [...prev];
        const lastWorkflow = [...updated].reverse().find(
          (m) => m.role === 'assistant' && m.workflowState
        );
        if (lastWorkflow) {
          lastWorkflow.workflowState = {
            ...lastWorkflow.workflowState,
            execution_result: data.execution,
            charts: data.chart ? [data.chart] : [],
          };
        }
        // Remove loading message
        return updated.filter((m) => m.content !== `execute-${algorithm}`);
      });
    } catch (err) {
      setMessages((prev) =>
        prev.filter((m) => m.content !== `execute-${algorithm}`).concat([
          { role: 'assistant', content: 'execute-error', error: err.message },
        ])
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSendWorkflow = async (query) => {
    const userMsg = { role: 'user', content: query };
    const loadingMsg = {
      role: 'assistant',
      content: 'workflow-loading',
      loading: true,
      stepResults: {},
      currentStep: 'STEP_1_PROBLEM',
    };
    setMessages((prev) => [...prev, userMsg, loadingMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch(`${API_WORKFLOW}/workflow/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();

      setMessages((prev) => {
        const updated = [...prev];
        updated.pop(); // remove loading message
        updated.push({
          role: 'assistant',
          content: query,
          workflowState: data.state,
          elapsedMs: data.elapsed_ms,
        });
        return updated;
      });
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        updated.pop();
        updated.push({ role: 'assistant', content: 'error', error: err.message });
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSendQuick = async (query) => {
    const userMsg = { role: 'user', content: query };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch(`${API_RECOMMEND}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 5 }),
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
        { role: 'assistant', content: 'error', error: err.message },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSend = () => {
    const query = input.trim();
    if (!query || loading) return;

    if (mode === 'workflow') {
      handleSendWorkflow(query);
    } else {
      handleSendQuick(query);
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
        <div className="mode-switch">
          <button
            className={mode === 'workflow' ? 'mode-active' : ''}
            onClick={() => setMode('workflow')}
          >
            7-Step Workflow
          </button>
          <button
            className={mode === 'quick' ? 'mode-active' : ''}
            onClick={() => setMode('quick')}
          >
            Quick Recommend
          </button>
        </div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <h2>Solver Agent</h2>
            <p>
              {mode === 'workflow'
                ? 'Describe your optimization problem. The agent will analyze it through a 7-step pipeline: problem understanding → dataset selection → variables → objectives → constraints → classification → algorithm recommendation.'
                : 'Quick RAG-based algorithm recommendation from the knowledge base.'}
            </p>
            <div className="example-queries">
              <span>Try:</span>
              {[
                '5台机器，20个工件，最小化完工时间',
                '多目标生产调度推荐什么算法？',
                '10辆车50个客户的有容量约束路径规划',
              ].map((q) => (
                <button key={q} onClick={() => setInput(q)}>{q}</button>
              ))}
            </div>
          </div>
        )}
        {messages.map((msg, i) => (
          <Message key={i} msg={msg} onExecute={handleExecute} />
        ))}
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
          {loading ? 'Running...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
