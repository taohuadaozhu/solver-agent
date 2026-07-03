const elements = {
  searchInput: document.getElementById("searchInput"),
  problemTypeSelect: document.getElementById("problemTypeSelect"),
  objectiveTypeSelect: document.getElementById("objectiveTypeSelect"),
  variableTypeSelect: document.getElementById("variableTypeSelect"),
  algorithmFamilySelect: document.getElementById("algorithmFamilySelect"),
  resultCount: document.getElementById("resultCount"),
  algorithmList: document.getElementById("algorithmList"),
};

let algorithms = [];

function normalize(text) {
  return String(text || "").toLowerCase();
}

function updateResultCount(count, total) {
  elements.resultCount.textContent = `${count} / ${total} 条算法结果`;
}

function createOption(select, values) {
  const existing = new Set(
    Array.from(select.options).map((item) => item.value),
  );
  values.sort();
  for (const value of values) {
    if (!value || existing.has(value)) continue;
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }
}

function getUniqueValues(items, key) {
  const result = new Set();
  items.forEach((item) => {
    const values = item.metadata[key];
    if (Array.isArray(values)) {
      values.forEach((value) => result.add(value));
    }
  });
  return Array.from(result);
}

function renderFilters() {
  createOption(
    elements.problemTypeSelect,
    getUniqueValues(algorithms, "problemType"),
  );
  createOption(
    elements.objectiveTypeSelect,
    getUniqueValues(algorithms, "objectiveType"),
  );
  createOption(
    elements.variableTypeSelect,
    getUniqueValues(algorithms, "variableType"),
  );
  createOption(
    elements.algorithmFamilySelect,
    getUniqueValues(algorithms, "algorithmFamily"),
  );
}

function renderCard(item) {
  const card = document.createElement("article");
  card.className = "algorithm-card";

  const title = document.createElement("h2");
  title.textContent = item.name || item.id;
  card.appendChild(title);

  const description = document.createElement("p");
  description.textContent =
    item.description || item.metadata.description || "暂无描述。";
  card.appendChild(description);

  const tags = document.createElement("div");
  tags.className = "tag-list";

  ["problemType", "objectiveType", "variableType", "algorithmFamily"].forEach(
    (key) => {
      const values = item.metadata[key];
      if (Array.isArray(values)) {
        values.forEach((value) => {
          const tag = document.createElement("span");
          tag.className = "tag";
          tag.textContent = value;
          tags.appendChild(tag);
        });
      }
    },
  );

  card.appendChild(tags);

  const detailLink = document.createElement("a");
  detailLink.href = `../knowledge-base/algorithms/${item.path}`;
  detailLink.textContent = "打开算法目录";
  detailLink.target = "_blank";
  detailLink.style.marginTop = "auto";
  detailLink.style.color = "#2563eb";
  detailLink.style.textDecoration = "none";
  card.appendChild(detailLink);

  return card;
}

function filterAlgorithms() {
  const query = normalize(elements.searchInput.value);
  const problemType = elements.problemTypeSelect.value;
  const objectiveType = elements.objectiveTypeSelect.value;
  const variableType = elements.variableTypeSelect.value;
  const algorithmFamily = elements.algorithmFamilySelect.value;

  return algorithms.filter((item) => {
    const text = [
      item.name,
      item.description,
      item.metadata.name,
      item.metadata.description,
    ]
      .filter(Boolean)
      .join(" ");

    if (query && !normalize(text).includes(query)) {
      return false;
    }

    if (
      problemType &&
      !(item.metadata.problemType || []).includes(problemType)
    ) {
      return false;
    }
    if (
      objectiveType &&
      !(item.metadata.objectiveType || []).includes(objectiveType)
    ) {
      return false;
    }
    if (
      variableType &&
      !(item.metadata.variableType || []).includes(variableType)
    ) {
      return false;
    }
    if (
      algorithmFamily &&
      !(item.metadata.algorithmFamily || []).includes(algorithmFamily)
    ) {
      return false;
    }

    return true;
  });
}

function renderList() {
  const filtered = filterAlgorithms();
  elements.algorithmList.innerHTML = "";

  if (!filtered.length) {
    const empty = document.createElement("div");
    empty.className = "no-results";
    empty.textContent = "未找到匹配的算法，请修改筛选条件。";
    elements.algorithmList.appendChild(empty);
    updateResultCount(0, algorithms.length);
    return;
  }

  filtered.forEach((item) => {
    elements.algorithmList.appendChild(renderCard(item));
  });

  updateResultCount(filtered.length, algorithms.length);
}

function bindEvents() {
  [
    elements.searchInput,
    elements.problemTypeSelect,
    elements.objectiveTypeSelect,
    elements.variableTypeSelect,
    elements.algorithmFamilySelect,
  ].forEach((element) => {
    element.addEventListener("input", renderList);
  });
}

async function init() {
  try {
    const response = await fetch("data/algorithms.json");
    algorithms = await response.json();
  } catch (error) {
    elements.algorithmList.innerHTML =
      '<div class="no-results">无法加载算法数据，请先运行 generate_index.py 生成 data/algorithms.json。</div>';
    elements.resultCount.textContent = "";
    console.error(error);
    return;
  }

  renderFilters();
  bindEvents();
  renderList();
}

init();
