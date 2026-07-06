import { useEffect, useMemo, useState } from 'react';
import ChatPage from './ChatPage';

const algorithmFilterKeys = [
  { key: 'problemType', label: '问题类型' },
  { key: 'objectiveType', label: '目标类型' },
  { key: 'variableType', label: '变量类型' },
  { key: 'algorithmFamily', label: '算法家族' },
];

const projectFilterKeys = [
  { key: 'domain', label: '领域' },
  { key: 'tags', label: '标签' },
];

const routes = [
  { key: 'algorithms', label: '算法' },
  { key: 'projects', label: '案例' },
  { key: 'chat', label: '智能推荐' },
];

function groupValues(items, key) {
  const values = items.flatMap((item) => {
    const value = item.metadata?.[key];
    if (Array.isArray(value)) return value;
    if (typeof value === 'string' && value.trim()) return [value];
    return [];
  });
  return [...new Set(values)].sort();
}

function FilterSelect({ label, value, options, onChange }) {
  return (
    <div className="control-item">
      <label>{label}</label>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">全部</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  );
}

function AlgorithmCard({ algorithm }) {
  return (
    <article className="algorithm-card">
      <div>
        <h2>{algorithm.name}</h2>
        <p>{algorithm.description || algorithm.metadata?.description || '暂无描述。'}</p>
      </div>
      <div className="tag-list">
        {algorithmFilterKeys.map(({ key }) =>
          (algorithm.metadata?.[key] || []).map((value) => (
            <span key={`${algorithm.id}-${key}-${value}`} className="tag">
              {value}
            </span>
          )),
        )}
      </div>
      <a className="detail-link" href={`../knowledge-base/algorithms/${algorithm.path}`} target="_blank" rel="noreferrer">
        打开算法目录
      </a>
    </article>
  );
}

function ProjectCard({ project }) {
  return (
    <article className="algorithm-card">
      <div>
        <h2>{project.name}</h2>
        <p>{project.description || project.metadata?.overview || '暂无描述。'}</p>
      </div>
      <div className="tag-list">
        {project.metadata?.domain && <span className="tag">{project.metadata.domain}</span>}
        {(project.metadata?.tags || []).map((tag) => (
          <span key={`${project.id}-${tag}`} className="tag">
            {tag}
          </span>
        ))}
        {(project.metadata?.recommendedAlgorithms || []).map((algo) => (
          <span key={`${project.id}-algo-${algo}`} className="tag">
            {algo}
          </span>
        ))}
      </div>
      <a className="detail-link" href={`../knowledge-base/projects/${project.path}`} target="_blank" rel="noreferrer">
        打开案例目录
      </a>
    </article>
  );
}

function App() {
  const [page, setPage] = useState('algorithms');
  const [algorithms, setAlgorithms] = useState([]);
  const [projects, setProjects] = useState([]);
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState({
    problemType: '',
    objectiveType: '',
    variableType: '',
    algorithmFamily: '',
  });
  const [projectFilters, setProjectFilters] = useState({
    domain: '',
    tags: '',
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const hashPage = window.location.hash.replace('#/', '') || 'algorithms';
    if (routes.some((route) => route.key === hashPage)) {
      setPage(hashPage);
    }
    const handleHashChange = () => {
      const nextPage = window.location.hash.replace('#/', '') || 'algorithms';
      if (routes.some((route) => route.key === nextPage)) {
        setPage(nextPage);
      }
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  useEffect(() => {
    Promise.all([
      fetch('/data/algorithms.json').then((res) => {
        if (!res.ok) throw new Error('无法加载算法数据');
        return res.json();
      }),
      fetch('/data/projects.json').then((res) => {
        if (!res.ok) throw new Error('无法加载案例数据');
        return res.json();
      }),
    ])
      .then(([algorithmsData, projectsData]) => {
        setAlgorithms(algorithmsData);
        setProjects(projectsData);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const normalizedQuery = query.trim().toLowerCase();

  const filteredAlgorithms = useMemo(() => {
    return algorithms.filter((item) => {
      const text = [item.name, item.description, item.metadata?.name, item.metadata?.description]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
      if (normalizedQuery && !text.includes(normalizedQuery)) {
        return false;
      }
      return algorithmFilterKeys.every(({ key }) => {
        if (!filters[key]) return true;
        return (item.metadata?.[key] || []).includes(filters[key]);
      });
    });
  }, [algorithms, filters, normalizedQuery]);

  const filteredProjects = useMemo(() => {
    return projects.filter((item) => {
      const text = [
        item.name,
        item.description,
        item.metadata?.overview,
        item.metadata?.businessScenarios?.join(' '),
        item.metadata?.recommendedAlgorithms?.join(' '),
        item.metadata?.objective?.join(' '),
        item.metadata?.constraints?.join(' '),
        item.metadata?.decisionVariables?.join(' '),
        item.metadata?.tags?.join(' '),
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();

      if (normalizedQuery && !text.includes(normalizedQuery)) {
        return false;
      }
      if (projectFilters.domain && item.metadata?.domain !== projectFilters.domain) {
        return false;
      }
      if (projectFilters.tags && !(item.metadata?.tags || []).includes(projectFilters.tags)) {
        return false;
      }
      return true;
    });
  }, [projects, normalizedQuery, projectFilters]);

  const filterOptions = useMemo(() => {
    return {
      algorithm: algorithmFilterKeys.reduce((acc, { key }) => {
        acc[key] = groupValues(algorithms, key);
        return acc;
      }, {}),
      project: projectFilterKeys.reduce((acc, { key }) => {
        acc[key] = groupValues(projects, key);
        return acc;
      }, {}),
    };
  }, [algorithms, projects]);

  const handleRouteClick = (route) => {
    window.location.hash = `#/${route}`;
    setPage(route);
  };

  return (
    <div className="app-shell">
      <header>
        <h1>Solver Agent</h1>
        <p>基于 knowledge-base 的算法推荐与案例检索平台。</p>
      </header>

      <nav className="page-nav">
        {routes.map((route) => (
          <button
            key={route.key}
            className={page === route.key ? 'active' : ''}
            onClick={() => handleRouteClick(route.key)}
          >
            {route.label}
          </button>
        ))}
      </nav>

      {page === 'chat' ? (
        <ChatPage />
      ) : (
        <>
          <section className="controls">
            <div className="control-item">
              <label>关键词搜索</label>
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="输入算法、案例名或描述关键字"
              />
            </div>
            {page === 'algorithms' &&
              algorithmFilterKeys.map(({ key, label }) => (
                <FilterSelect
                  key={key}
                  label={label}
                  value={filters[key]}
                  options={filterOptions.algorithm[key] || []}
                  onChange={(value) => setFilters((prev) => ({ ...prev, [key]: value }))}
                />
              ))}
            {page === 'projects' &&
              projectFilterKeys.map(({ key, label }) => (
                <FilterSelect
                  key={key}
                  label={label}
                  value={projectFilters[key]}
                  options={filterOptions.project[key] || []}
                  onChange={(value) => setProjectFilters((prev) => ({ ...prev, [key]: value }))}
                />
              ))}
          </section>

          <section className="result-count">
            {page === 'algorithms'
              ? `${filteredAlgorithms.length} / ${algorithms.length} 条算法结果`
              : `${filteredProjects.length} / ${projects.length} 条案例结果`}
          </section>

          <section className="algorithm-list">
            {loading && <div className="no-results">正在加载数据...</div>}
            {error && <div className="no-results">{error}</div>}

            {!loading && !error && page === 'algorithms' && filteredAlgorithms.length === 0 && (
              <div className="no-results">未找到匹配的算法，请修改筛选条件。</div>
            )}
            {!loading && !error && page === 'projects' && filteredProjects.length === 0 && (
              <div className="no-results">未找到匹配的案例，请修改搜索关键词。</div>
            )}

            {!loading && !error && page === 'algorithms' && filteredAlgorithms.map((algorithm) => (
              <AlgorithmCard key={algorithm.id} algorithm={algorithm} />
            ))}
            {!loading && !error && page === 'projects' && filteredProjects.map((project) => (
              <ProjectCard key={project.id} project={project} />
            ))}
          </section>
        </>
      )}
    </div>
  );
}

export default App;
