let missions = [];
let config = { sage_ground_data_path: "../sage_ground_data" };
let importedImageUrls = [];

const sagePaletteLight = {
  bg: "#cbd2c5",
  grid: "#b8caa1",
  gridSoft: "#c5d5b0",
  surface: "#dfe5da",
  surfaceActive: "#b8caa1",
  accent: "#617c32",
  text: "#344f41",
  muted: "#697d6a"
};

const sagePaletteDark = {
  bg: "#10171a",
  grid: "#244234",
  gridSoft: "#355545",
  surface: "#122018",
  surfaceActive: "#1e3126",
  accent: "#66D17A",
  text: "#d4dee7",
  muted: "#9db4c4"
};

function getSagePalette() {
  const theme = document.documentElement.getAttribute("data-theme") || "light";
  return theme === "light" ? sagePaletteLight : sagePaletteDark;
}

// For backward compatibility
const sagePalette = sagePaletteLight;

const missionSelectEl = document.getElementById("missionSelect");
const missionTreeEl = document.getElementById("missionTree");
const sampleSearchEl = document.getElementById("sampleSearch");
const classificationFilterEl = document.getElementById("classificationFilter");
const volumeFilterEl = document.getElementById("volumeFilter");
const dateFilterEl = document.getElementById("dateFilter");
const viewerModeLabelEl = document.getElementById("viewerModeLabel");
const sampleViewSectionEl = document.getElementById("sampleViewSection");
const audioViewSectionEl = document.getElementById("audioViewSection");
const sampleInfoStripEl = document.getElementById("sampleInfoStrip");
const topViewImageEl = document.getElementById("topViewImage");
const sideViewImageEl = document.getElementById("sideViewImage");
const sampleMetadataGridEl = document.getElementById("sampleMetadataGrid");
const transcriptMetadataGridEl = document.getElementById("transcriptMetadataGrid");
const classificationGridEl = document.getElementById("classificationGrid");
const sampleStripEl = document.getElementById("sampleStrip");
const stripSortEl = document.getElementById("stripSort");
const waveformEl = document.getElementById("waveform");
const audioListEl = document.getElementById("audioList");
const audioTimestampEl = document.getElementById("audioTimestamp");
const importDatasetBtnEl = document.getElementById("importDatasetBtn");
const datasetFileInputEl = document.getElementById("datasetFileInput");
const themeToggleBtnEl = document.getElementById("themeToggleBtn");

// Initialize theme from localStorage
function initializeTheme() {
  const savedTheme = localStorage.getItem("theme") || "light";
  document.documentElement.setAttribute("data-theme", savedTheme);
  updateThemeButton(savedTheme);
}

function updateThemeButton(theme) {
  if (themeToggleBtnEl) {
    themeToggleBtnEl.textContent = theme === "light" ? "🌙" : "☀️";
  }
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
  const newTheme = currentTheme === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", newTheme);
  localStorage.setItem("theme", newTheme);
  updateThemeButton(newTheme);
}

if (themeToggleBtnEl) {
  themeToggleBtnEl.addEventListener("click", toggleTheme);
}

initializeTheme();

const zoomState = {
  top: 1,
  side: 1
};

const uiState = {
  missionId: "",
  selectedType: "sample",
  selectedSampleId: "",
  selectedAudioId: "",
  selectedTranscriptId: "",
  missionExpanded: true,
  audioExpanded: true,
  transcriptsExpanded: true,
  search: "",
  classFilter: "all",
  volumeFilter: "all",
  dateFilter: "all",
  sortBy: "timestamp"
};

function utcLabel(utc) {
  return utc.replace("T", " ").replace("Z", " UTC");
}

function getTranscriptPreview(text, maxLength = 40) {
  if (!text) return "[empty]";
  const words = text.trim().split(/\s+/).slice(0, 6);
  let preview = words.join(" ");
  if (preview.length > maxLength) {
    preview = preview.substring(0, maxLength) + "...";
  } else if (preview.length < text.trim().length) {
    preview += "...";
  }
  return preview;
}

function placeholderImage(label, sampleId) {
  const palette = getSagePalette();
  const svg = `
    <svg xmlns='http://www.w3.org/2000/svg' width='800' height='600'>
      <rect width='100%' height='100%' fill='${palette.bg}'/>
      <g stroke='${palette.grid}' stroke-width='1'>
        <line x1='0' y1='80' x2='800' y2='80'/>
        <line x1='0' y1='200' x2='800' y2='200'/>
        <line x1='0' y1='320' x2='800' y2='320'/>
        <line x1='0' y1='440' x2='800' y2='440'/>
        <line x1='120' y1='0' x2='120' y2='600'/>
        <line x1='280' y1='0' x2='280' y2='600'/>
        <line x1='440' y1='0' x2='440' y2='600'/>
        <line x1='600' y1='0' x2='600' y2='600'/>
      </g>
      <rect x='250' y='170' width='300' height='240' fill='none' stroke='${palette.accent}' stroke-width='2'/>
      <text x='400' y='85' font-size='30' fill='${palette.accent}' text-anchor='middle' font-family='IBM Plex Mono, Consolas, monospace'>${label}</text>
      <text x='400' y='535' font-size='24' fill='${palette.text}' text-anchor='middle' font-family='IBM Plex Mono, Consolas, monospace'>${sampleId}</text>
    </svg>
  `;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function blankImage() {
  return "data:image/svg+xml;charset=utf-8," + encodeURIComponent("<svg xmlns='http://www.w3.org/2000/svg' width='800' height='600'><rect width='100%' height='100%' fill='transparent'/></svg>");
}

function clearImportedImageUrls() {
  importedImageUrls.forEach((url) => URL.revokeObjectURL(url));
  importedImageUrls = [];
}

function parseJsonl(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function makeObjectUrl(file) {
  const url = URL.createObjectURL(file);
  importedImageUrls.push(url);
  return url;
}

function hasClassificationData(sample) {
  const classification = String(sample?.classification ?? "").trim();
  return Boolean(classification) && classification.toLowerCase() !== "unknown";
}

function hasMatchedImagePair(sample) {
  return Boolean(sample?.hasImagePair) && Boolean(sample?.topImagePath || sample?.imagePath) && Boolean(sample?.sideImagePath || sample?.imagePath);
}

function isDisplayableSample(sample) {
  return hasClassificationData(sample) && hasMatchedImagePair(sample);
}

function normalizeMissionsForDisplay(missionList) {
  return (missionList || [])
    .map((mission) => ({
      ...mission,
      samples: (mission.samples || []).filter(isDisplayableSample),
      audioFiles: mission.audioFiles || [],
    }))
    .filter((mission) => mission.samples.length > 0);
}

async function parseFolderImport(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) {
    return [];
  }

  const fileMap = new Map();
  files.forEach((file) => {
    const key = file.webkitRelativePath || file.name;
    fileMap.set(key.replace(/^\/?/, ""), file);
  });

  const missionsFiles = files
    .filter((file) => file.name === "missions.json")
    .sort((a, b) => (a.webkitRelativePath || a.name).split(/[\\/]/).length - (b.webkitRelativePath || b.name).split(/[\\/]/).length);

  for (const missionsFile of missionsFiles) {
    const data = JSON.parse(await missionsFile.text());
    const missionsList = Array.isArray(data) ? data : Object.values(data);
    if (missionsList.length && missionsList.every((mission) => Array.isArray(mission?.samples))) {
      return normalizeMissionsForDisplay(missionsList);
    }
  }

  const rocksEntries = [...fileMap.entries()].filter(([path]) => path.endsWith("/sage_store/rocks.jsonl"));
  if (!rocksEntries.length) {
    throw new Error("Selected folder does not contain missions.json or sage_store/rocks.jsonl");
  }

  const missionsById = new Map();

  for (const [rocksPath, rocksFile] of rocksEntries) {
    const runRoot = rocksPath.slice(0, -"/sage_store/rocks.jsonl".length);
    const voicePath = `${runRoot}/sage_store/voice_notes.jsonl`;
    const voiceFile = fileMap.get(voicePath);
    const imagePrefix = `${runRoot}/images/`;

    const imagesByFilename = new Map();
    for (const [path, file] of fileMap.entries()) {
      if (!path.startsWith(imagePrefix) || !path.toLowerCase().endsWith(".jpg")) {
        continue;
      }
      const filename = path.split("/").pop();
      if (filename) {
        imagesByFilename.set(filename, makeObjectUrl(file));
      }
    }

    const rocks = parseJsonl(await rocksFile.text()).reduce((acc, obj) => {
      const rockId = obj.rock_id;
      acc.set(rockId, {
        id: rockId.slice(0, 12),
        full_id: rockId,
        timestamp: obj.ts || 0,
        classification: obj?.result?.label || "Unknown",
        confidence: Number(obj?.result?.confidence || 0),
        volume: obj?.result?.estimated_volume,
        topImagePath: obj?.result?.image_path || null,
        sideImagePath: obj?.result?.side_image_path || null,
      });
      return acc;
    }, new Map());

    const voiceNotes = new Map();
    const unmatchedVoiceNotes = [];
    if (voiceFile) {
      for (const obj of parseJsonl(await voiceFile.text())) {
        const rockId = obj.rock_id;
        const note = {
          timestamp: obj.ts || 0,
          transcript: obj.transcript || "",
          session_id: obj.session_id,
        };
        if (rocks.has(rockId)) {
          if (!voiceNotes.has(rockId)) {
            voiceNotes.set(rockId, []);
          }
          voiceNotes.get(rockId).push(note);
        } else {
          unmatchedVoiceNotes.push(note);
        }
      }
    }

    const runFolderName = runRoot.split("/").filter(Boolean).pop() || runRoot;
    const missionId = runFolderName;
    const rockIds = [...rocks.values()]
      .sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0))
      .map((rock) => rock.full_id);

    if (unmatchedVoiceNotes.length && rockIds.length) {
      const rockTimestamps = new Map(rockIds.map((rockId) => [rockId, rocks.get(rockId)?.timestamp || 0]));
      unmatchedVoiceNotes.forEach((note) => {
        const nearestRockId = rockIds.reduce((bestId, rockId) => {
          if (!bestId) return rockId;
          const bestDelta = Math.abs((rockTimestamps.get(bestId) || 0) - note.timestamp);
          const currentDelta = Math.abs((rockTimestamps.get(rockId) || 0) - note.timestamp);
          return currentDelta < bestDelta ? rockId : bestId;
        }, "");
        if (nearestRockId && Math.abs((rockTimestamps.get(nearestRockId) || 0) - note.timestamp) < 10800) {
          if (!voiceNotes.has(nearestRockId)) {
            voiceNotes.set(nearestRockId, []);
          }
            voiceNotes.get(nearestRockId).push(note);
            // mark this unmatched note as assigned so we can expose remaining unlinked notes
            try { note._assigned = true; } catch (e) { /* noop */ }
        }
      });
    }

    const samples = rockIds.map((rockId) => {
      const rock = rocks.get(rockId);
      const timestampUtc = new Date((rock.timestamp || 0) * 1000).toISOString();
      const classification = String(rock.classification || "").trim();
      const topFile = rock.topImagePath ? rock.topImagePath.split("/").pop() : null;
      const sideFile = rock.sideImagePath ? rock.sideImagePath.split("/").pop() : null;
      const topImagePath = topFile ? imagesByFilename.get(topFile) : null;
      const sideImagePath = sideFile ? imagesByFilename.get(sideFile) : null;

      if (!classification || classification.toLowerCase() === "unknown" || !topImagePath || !sideImagePath) {
        return null;
      }

      return {
        id: rock.id,
        full_id: rock.full_id,
        timestampUtc,
        classification,
        confidence: rock.confidence,
        predictionTimestamp: timestampUtc,
        volumeCm3: rock.volume || 0,
        dimensions: { length: 0, width: 0, height: 0 },
        footprintArea: 0,
        shapeFactor: 0,
        estimationMethod: "Stored metadata",
        scaleTop: 0.12,
        scaleSide: 0.11,
        hasImagePair: true,
        imagePath: topImagePath,
        topImagePath,
        sideImagePath,
        notes: "",
        voiceNotes: voiceNotes.get(rock.full_id) || [],
      };
    }).filter(Boolean);

    if (samples.length) {
      missionsById.set(missionId, {
        id: missionId,
        operator: "Ground Analyst",
        samples,
        audioFiles: [],
        unlinkedVoiceNotes: unmatchedVoiceNotes.filter((n) => !n._assigned),
      });
    }
  }

  return normalizeMissionsForDisplay([...missionsById.values()]);
}

function getImageCandidates(sample, pathKey = "imagePath") {
  const sourcePath = sample?.[pathKey] || sample?.imagePath;
  if (!sourcePath) {
    return [];
  }

  const rawPath = String(sourcePath).replace(/^\.\//, "");
  if (/^(blob:|data:|file:|https?:)/i.test(rawPath)) {
    return [rawPath];
  }

  const basePath = String(config.sage_ground_data_path || "../sage_ground_data").replace(/\\/g, "/");
  const candidates = [];

  if (/^[A-Z]:/i.test(basePath)) {
    candidates.push(`file:///${basePath}/${rawPath}`);
    candidates.push(`../sage_ground_data/${rawPath}`);
  } else {
    candidates.push(`${basePath}/${rawPath}`);
    if (!basePath.includes("sage_ground_data")) {
      candidates.push(`../sage_ground_data/${rawPath}`);
    }
  }

  return [...new Set(candidates)];
}

function getImagePath(sample, includeParent = true) {
  const candidates = getImageCandidates(sample);
  return candidates[0] || null;
}

function setImageWithFallback(imageEl, candidates, fallbackSrc) {
  if (!candidates.length) {
    imageEl.onerror = null;
    imageEl.src = fallbackSrc;
    return;
  }

  let index = 0;
  imageEl.onerror = () => {
    index += 1;
    if (index < candidates.length) {
      imageEl.src = candidates[index];
      return;
    }
    imageEl.onerror = null;
    imageEl.src = fallbackSrc;
  };
  imageEl.src = candidates[0];
}

async function loadMissionsData() {
  try {
    // Load config first
    try {
      const configResponse = await fetch('./config.json');
      if (configResponse.ok) {
        config = await configResponse.json();
      }
    } catch (err) {
      console.warn('Could not load config.json, using default path:', err);
    }
    
    // Load missions
    const response = await fetch('./missions.json');
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    missions = normalizeMissionsForDisplay(Object.values(data));
    return true;
  } catch (err) {
    console.error('Failed to load missions.json:', err);
    return false;
  }
}

function getMission() {
  return missions.find((m) => m.id === uiState.missionId) || missions[0];
}

function getSample() {
  const mission = getMission();
  if (!mission?.samples?.length) {
    return null;
  }
  return mission.samples.find((s) => s.id === uiState.selectedSampleId) || mission.samples[0];
}

function getDefaultSampleIdForMission(mission) {
  if (!mission?.samples?.length) {
    return "";
  }
  return mission.samples[0].id;
}

function formatClassificationLabel(raw) {
  const s = String(raw || "Unknown").trim();
  if (!s) return "Unknown";
  return s.toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
}

function getDefaultTranscriptIdForMission(mission) {
  if (!mission?.samples?.length) {
    return "";
  }

  for (const sample of mission.samples) {
    if (sample.voiceNotes && sample.voiceNotes.length > 0) {
      return `${sample.full_id}_0`;
    }
  }

  // If no sample-attached transcripts, but there are unlinked voice notes, pick the first
  if (mission.unlinkedVoiceNotes && mission.unlinkedVoiceNotes.length > 0) {
    return `unlinked_${mission.id}_0`;
  }

  return "";
}

function getAudio() {
  const mission = getMission();
  return mission.audioFiles.find((a) => a.id === uiState.selectedAudioId) || mission.audioFiles[0];
}

function getAllTranscripts() {
  const mission = getMission();
  const transcripts = [];
  mission.samples.forEach((sample) => {
    if (sample.voiceNotes && sample.voiceNotes.length > 0) {
      sample.voiceNotes.forEach((note, idx) => {
        transcripts.push({
          id: `${sample.full_id}_${idx}`,
          sampleId: sample.id,
          sampleFullId: sample.full_id,
          timestamp: note.timestamp,
          timestampUtc: new Date(note.timestamp * 1000).toISOString(),
          transcript: note.transcript,
          sessionId: note.session_id
        });
      });
    }
  });
  // include unlinked voice notes (not attached to any sample)
  if (mission.unlinkedVoiceNotes && mission.unlinkedVoiceNotes.length > 0) {
    mission.unlinkedVoiceNotes.forEach((note, idx) => {
      transcripts.push({
        id: `unlinked_${mission.id}_${idx}`,
        sampleId: null,
        sampleFullId: null,
        timestamp: note.timestamp,
        timestampUtc: new Date((note.timestamp || 0) * 1000).toISOString(),
        transcript: note.transcript,
        sessionId: note.session_id,
        unlinked: true
      });
    });
  }
  return transcripts.sort((a, b) => a.timestamp - b.timestamp);
}

function getTranscript() {
  const transcripts = getAllTranscripts();
  return transcripts.find((t) => t.id === uiState.selectedTranscriptId) || null;
}

function getSelectedTranscriptForSample(sample) {
  const transcripts = getAllTranscripts();
  if (!transcripts.length) {
    return null;
  }

  const selectedTranscript = transcripts.find((t) => t.id === uiState.selectedTranscriptId);
  if (selectedTranscript && selectedTranscript.sampleId === sample.id) {
    return selectedTranscript;
  }

  return transcripts.find((t) => t.sampleId === sample.id) || null;
}

function initMissionSelector() {
  missionSelectEl.innerHTML = "";
  missions.forEach((mission) => {
    const opt = document.createElement("option");
    opt.value = mission.id;
    opt.textContent = mission.id;
    missionSelectEl.appendChild(opt);
  });

  missionSelectEl.value = uiState.missionId;

  missionSelectEl.addEventListener("change", () => {
    uiState.missionId = missionSelectEl.value;
    const mission = getMission();
    uiState.selectedSampleId = getDefaultSampleIdForMission(mission);
    uiState.selectedTranscriptId = getDefaultTranscriptIdForMission(mission);
    uiState.selectedAudioId = mission.audioFiles[0]?.id || "";
    uiState.selectedType = "sample";
    refreshFilters();
    render();
  });
}

function getFilteredSamples() {
  const mission = getMission();
  if (!mission?.samples?.length) {
    return [];
  }
  return mission.samples
    .filter((sample) => {
    if (!isDisplayableSample(sample)) {
      return false;
    }

    const normalizedSearch = uiState.search.trim().toLowerCase();
    const searchMatch =
      !normalizedSearch ||
      sample.id.toLowerCase().includes(normalizedSearch) ||
      sample.timestampUtc.toLowerCase().includes(normalizedSearch);

    const classMatch = uiState.classFilter === "all" || formatClassificationLabel(sample.classification || "Unknown") === uiState.classFilter;

    let volumeMatch = true;
    if (uiState.volumeFilter === "small") {
      volumeMatch = sample.volumeCm3 <= 120;
    } else if (uiState.volumeFilter === "medium") {
      volumeMatch = sample.volumeCm3 > 120 && sample.volumeCm3 <= 220;
    } else if (uiState.volumeFilter === "large") {
      volumeMatch = sample.volumeCm3 > 220;
    }

    const datePrefix = sample.timestampUtc.slice(0, 10);
    const dateMatch = uiState.dateFilter === "all" || datePrefix === uiState.dateFilter;

    return searchMatch && classMatch && volumeMatch && dateMatch;
    })
    .sort((a, b) => a.timestampUtc.localeCompare(b.timestampUtc));
}

function getSampleStatus(sample) {
  if (sample.hasImagePair) {
    return { label: "ok", isWarn: false };
  }

  if (sample.imagePath) {
    const filename = String(sample.imagePath).split("/").pop() || "";
    const currentPattern = /^\d{8}_\d{6}_[0-9a-fA-F-]+\.jpg$/i;
    if (!currentPattern.test(filename)) {
      return { label: "outdated", isWarn: true };
    }
  }

  return { label: "missing pair", isWarn: true };
}

function getClassificationLabel(samples, targetSample) {
  const classificationCounts = {};
  const sampleOrder = [];

  samples.forEach((sample) => {
    const cls = formatClassificationLabel(sample.classification || "Unknown");
    if (!classificationCounts[cls]) {
      classificationCounts[cls] = 0;
    }
    classificationCounts[cls]++;
    if (sample.id === targetSample.id) {
      sampleOrder.push(classificationCounts[cls]);
    }
  });

  const count = sampleOrder[0] || 1;
  return `${formatClassificationLabel(targetSample.classification || "Unknown")}${count}`;
}

function refreshFilters() {
  const mission = getMission();
  if (!mission?.samples?.length) {
    classificationFilterEl.innerHTML = `<option value="all">All</option>`;
    dateFilterEl.innerHTML = `<option value="all">All</option>`;
    classificationFilterEl.value = "all";
    dateFilterEl.value = "all";
    return;
  }
  const classes = [...new Set(mission.samples.map((s) => s.classification))];
  const dates = [...new Set(mission.samples.map((s) => s.timestampUtc.slice(0, 10)))];

  classificationFilterEl.innerHTML = `<option value="all">All</option>`;
  classes.forEach((cls) => {
    const label = formatClassificationLabel(cls || "Unknown");
    classificationFilterEl.innerHTML += `<option value="${label}">${label}</option>`;
  });

  dateFilterEl.innerHTML = `<option value="all">All</option>`;
  dates.forEach((date) => {
    dateFilterEl.innerHTML += `<option value="${date}">${date}</option>`;
  });

  classificationFilterEl.value = "all";
  dateFilterEl.value = "all";
}

function renderTree() {
  const mission = getMission();
  if (!mission) {
    missionTreeEl.innerHTML = "";
    return;
  }
  const samples = getFilteredSamples();

  missionTreeEl.innerHTML = "";

  const missionSection = document.createElement("div");
  missionSection.className = "tree-section";
  missionSection.innerHTML = `
    <button type="button" class="tree-toggle" id="missionToggle">
      ${uiState.missionExpanded ? "[-]" : "[+]"} ${mission.id}
    </button>
  `;
  missionTreeEl.appendChild(missionSection);

  const missionChildren = document.createElement("div");
  missionChildren.className = "tree-children";
  missionChildren.style.display = uiState.missionExpanded ? "grid" : "none";

  samples.forEach((sample) => {
    const status = getSampleStatus(sample);
    const item = document.createElement("div");
    item.className = `tree-item ${uiState.selectedType === "sample" && uiState.selectedSampleId === sample.id ? "active" : ""}`;
    const classLabel = getClassificationLabel(samples, sample);
    item.innerHTML = `
      <span class="tree-item-label">${classLabel}</span>
      <span class="status-tag ${status.isWarn ? "warn" : ""}">${status.label}</span>
    `;
    item.addEventListener("click", () => {
      uiState.selectedType = "sample";
      uiState.selectedSampleId = sample.id;
      render();
    });
    missionChildren.appendChild(item);
  });

  
  const transcriptSection = document.createElement("div");
  transcriptSection.className = "tree-section";
  transcriptSection.innerHTML = `
    <button type="button" class="tree-toggle" id="transcriptsToggle">
      ${uiState.transcriptsExpanded ? "[-]" : "[+]"} Voice Transcripts
    </button>
  `;

  const transcriptChildren = document.createElement("div");
  transcriptChildren.className = "tree-children";
  transcriptChildren.style.display = uiState.transcriptsExpanded ? "grid" : "none";

  const transcripts = getAllTranscripts();
  transcripts.forEach((transcript) => {
    const item = document.createElement("div");
    item.className = `tree-item ${uiState.selectedType === "transcript" && uiState.selectedTranscriptId === transcript.id ? "active" : ""}`;
    const preview = getTranscriptPreview(transcript.transcript);
    item.innerHTML = `
      <span class="tree-item-label" title="${transcript.transcript}">${preview}</span>
      <span class="status-tag">note</span>
    `;
    item.addEventListener("click", () => {
      uiState.selectedType = "transcript";
      uiState.selectedTranscriptId = transcript.id;
      render();
    });
    transcriptChildren.appendChild(item);
  });

  
  missionChildren.appendChild(transcriptSection);
  missionChildren.appendChild(transcriptChildren);
  missionSection.appendChild(missionChildren);

  missionTreeEl.querySelector("#missionToggle")?.addEventListener("click", () => {
    uiState.missionExpanded = !uiState.missionExpanded;
    renderTree();
  });

  

  missionTreeEl.querySelector("#transcriptsToggle")?.addEventListener("click", () => {
    uiState.transcriptsExpanded = !uiState.transcriptsExpanded;
    renderTree();
  });
}

function kvRows(targetEl, rows) {
  targetEl.innerHTML = "";
  rows.forEach((row) => {
    const div = document.createElement("div");
    div.className = "kv-row";
    div.innerHTML = `
      <span class="kv-key">${row.key}</span>
      <span class="kv-value ${row.numeric ? "numeric" : ""}">${row.value}</span>
    `;
    targetEl.appendChild(div);
  });
}

function renderSampleViewer() {
  const mission = getMission();
  const sample = getSample();
  if (!mission || !sample) {
    return;
  }
  const sampleStatus = getSampleStatus(sample);
  sampleViewSectionEl.classList.remove("hidden");
  sampleInfoStripEl.classList.add("sample-info-strip-sample");

  const topCandidates = getImageCandidates(sample, "topImagePath");
  const sideCandidates = getImageCandidates(sample, "sideImagePath");
  const fallbackCandidates = getImageCandidates(sample, "imagePath");
  setImageWithFallback(topViewImageEl, topCandidates.length ? topCandidates : fallbackCandidates, placeholderImage("TOP VIEW", `${mission.id} ${sample.id}`));
  if (sampleStatus.label === "missing pair") {
    setImageWithFallback(sideViewImageEl, [blankImage()], blankImage());
  } else {
    setImageWithFallback(sideViewImageEl, sideCandidates.length ? sideCandidates : fallbackCandidates, placeholderImage("SIDE VIEW", `${mission.id} ${sample.id}`));
  }

  // Reset per-sample visual state so previous interactions don't distort new images.
  zoomState.top = 1;
  zoomState.side = 1;
  topViewImageEl.style.transform = "scale(1)";
  sideViewImageEl.style.transform = "scale(1)";

  const topOpacityInput = document.getElementById("topOpacity");
  const sideOpacityInput = document.getElementById("sideOpacity");
  const topOverlayInput = document.getElementById("topOverlayToggle");
  const sideOverlayInput = document.getElementById("sideOverlayToggle");

  const topOpacity = Number(topOpacityInput?.value ?? 100) / 100;
  const sideOpacity = sampleStatus.label === "missing pair" ? 0 : Number(sideOpacityInput?.value ?? 100) / 100;
  topViewImageEl.style.opacity = String(topOpacity);
  sideViewImageEl.style.opacity = String(sideOpacity);
  topViewImageEl.style.filter = topOverlayInput?.checked ? "contrast(1.1) saturate(1.05)" : "none";
  sideViewImageEl.style.filter = sampleStatus.label === "missing pair" ? "none" : (sideOverlayInput?.checked ? "contrast(1.1) saturate(1.05)" : "none");

  sampleInfoStripEl.innerHTML = `
    <div class="strip-cell strip-cell-sample"><span class="strip-label">Sample ID</span><span class="strip-value strip-value-strong">${sample.id}</span></div>
    <div class="strip-cell strip-cell-sample"><span class="strip-label">Timestamp (UTC)</span><span class="strip-value mono">${utcLabel(sample.timestampUtc)}</span></div>
    <div class="strip-cell strip-cell-sample strip-cell-emphasis"><span class="strip-label">Classification</span><span class="strip-value strip-value-accent">${formatClassificationLabel(sample.classification)}</span></div>
    <div class="strip-cell strip-cell-sample strip-cell-emphasis"><span class="strip-label">Estimated Volume</span><span class="strip-value strip-value-accent numeric-emphasis">${sample.volumeCm3.toFixed(1)} cm3</span></div>
  `;

  kvRows(sampleMetadataGridEl, [
    { key: "Sample ID", value: sample.id },
    { key: "Timestamp (UTC)", value: utcLabel(sample.timestampUtc) }
  ]);

  kvRows(classificationGridEl, [
    { key: "Predicted Class", value: formatClassificationLabel(sample.classification) },
    { key: "Confidence Score", value: `${(sample.confidence * 100).toFixed(1)}%`, numeric: true }
  ]);

  const transcript = getSelectedTranscriptForSample(sample) || getTranscript();
  transcriptMetadataGridEl.innerHTML = "";
  const transcriptContentEl = document.getElementById("transcriptContent");
  const transcriptTextEl = transcriptContentEl?.querySelector(".transcript-text");

  if (transcript) {
    const sampleLabel = transcript.unlinked ? "Unlinked" : (transcript.sampleId || "");
    transcriptMetadataGridEl.innerHTML = `
      <div class="kv-row"><span class="kv-key">Sample</span><span class="kv-value">${sampleLabel}</span></div>
      <div class="kv-row"><span class="kv-key">Timestamp (UTC)</span><span class="kv-value">${utcLabel(transcript.timestampUtc)}</span></div>
`;
    transcriptContentEl?.classList.remove("hidden");
    if (transcriptTextEl) {
      const transcriptText = (transcript.transcript || "").trim();
      transcriptTextEl.textContent = transcriptText && transcriptText.toLowerCase() !== "no" ? transcriptText : "None found";
    }
  } else {
    transcriptMetadataGridEl.innerHTML = `
      <div class="kv-row"><span class="kv-key">Sample</span><span class="kv-value">-</span></div>
      <div class="kv-row"><span class="kv-key">Timestamp (UTC)</span><span class="kv-value">-</span></div>
    `;
    transcriptContentEl?.classList.add("hidden");
    if (transcriptTextEl) {
      transcriptTextEl.textContent = "None found";
    }
  }
}

function renderStrip() {
  const mission = getMission();
  if (!mission?.samples?.length) {
    sampleStripEl.innerHTML = "";
    return;
  }
  const samples = [...mission.samples].filter((sample) => isDisplayableSample(sample));
  if (uiState.sortBy === "timestamp") {
    samples.sort((a, b) => a.timestampUtc.localeCompare(b.timestampUtc));
  } else if (uiState.sortBy === "volume") {
    samples.sort((a, b) => b.volumeCm3 - a.volumeCm3);
  } else {
    samples.sort((a, b) => formatClassificationLabel(a.classification || "").localeCompare(formatClassificationLabel(b.classification || "")));
  }

  sampleStripEl.innerHTML = "";
  samples.forEach((sample) => {
    const card = document.createElement("article");
    card.className = `sample-card ${uiState.selectedType === "sample" && uiState.selectedSampleId === sample.id ? "active" : ""}`;
    card.innerHTML = `
      <img class="sample-thumb" alt="${sample.id} thumbnail" />
      <div class="sample-card-grid">
        <strong>${sample.id}</strong>
        <span>${formatClassificationLabel(sample.classification)}</span>
        <span class="numeric-emphasis">${sample.volumeCm3.toFixed(1)} cm3</span>
        <span class="mono">${utcLabel(sample.timestampUtc)}</span>
      </div>
    `;
    const thumbEl = card.querySelector(".sample-thumb");
    setImageWithFallback(thumbEl, getImageCandidates(sample), placeholderImage("THUMB", sample.id));
    card.addEventListener("click", () => {
      uiState.selectedType = "sample";
      uiState.selectedSampleId = sample.id;
      render();
    });
    sampleStripEl.appendChild(card);
  });
}

function bindControls() {
  importDatasetBtnEl?.addEventListener("click", () => {
    datasetFileInputEl?.click();
  });

  datasetFileInputEl?.addEventListener("change", async (event) => {
    const files = event.target.files;
    if (!files || !files.length) {
      return;
    }

    try {
      clearImportedImageUrls();
      missions = await parseFolderImport(files);
      if (!missions.length) {
        throw new Error("Selected folder did not contain a mission folder");
      }

      uiState.missionId = missions[0].id;
      uiState.selectedType = "sample";
      uiState.selectedSampleId = getDefaultSampleIdForMission(missions[0]);
      uiState.selectedTranscriptId = getDefaultTranscriptIdForMission(missions[0]);
      uiState.selectedAudioId = missions[0].audioFiles[0]?.id || "";
      initMissionSelector();
      refreshFilters();
      render();
    } catch (err) {
      console.error("Failed to import dataset:", err);
      alert(`Could not import dataset: ${err.message}`);
    } finally {
      event.target.value = "";
    }
  });

  sampleSearchEl.addEventListener("input", () => {
    uiState.search = sampleSearchEl.value;
    renderTree();
  });

  classificationFilterEl.addEventListener("change", () => {
    uiState.classFilter = classificationFilterEl.value;
    renderTree();
  });

  volumeFilterEl.addEventListener("change", () => {
    uiState.volumeFilter = volumeFilterEl.value;
    renderTree();
  });

  dateFilterEl.addEventListener("change", () => {
    uiState.dateFilter = dateFilterEl.value;
    renderTree();
  });

  stripSortEl.addEventListener("change", () => {
    uiState.sortBy = stripSortEl.value;
    renderStrip();
  });

  document.getElementById("topZoomIn")?.addEventListener("click", () => {
    zoomState.top = Math.min(zoomState.top + 0.1, 2.2);
    topViewImageEl.style.transform = `scale(${zoomState.top})`;
  });

  document.getElementById("topZoomOut")?.addEventListener("click", () => {
    zoomState.top = Math.max(zoomState.top - 0.1, 1);
    topViewImageEl.style.transform = `scale(${zoomState.top})`;
  });

  document.getElementById("sideZoomIn")?.addEventListener("click", () => {
    zoomState.side = Math.min(zoomState.side + 0.1, 2.2);
    sideViewImageEl.style.transform = `scale(${zoomState.side})`;
  });

  document.getElementById("sideZoomOut")?.addEventListener("click", () => {
    zoomState.side = Math.max(zoomState.side - 0.1, 1);
    sideViewImageEl.style.transform = `scale(${zoomState.side})`;
  });

  
}

function render() {
  renderTree();
  renderStrip();
  renderSampleViewer();
}

// Bind controls once on page load
bindControls();

// Load real missions data
async function init() {
  const loaded = await loadMissionsData();
  if (loaded && missions.length > 0) {
    uiState.missionId = missions[0].id;
    uiState.selectedSampleId = getDefaultSampleIdForMission(missions[0]);
    uiState.selectedTranscriptId = getDefaultTranscriptIdForMission(missions[0]);
    uiState.selectedAudioId = missions[0].audioFiles[0]?.id || "";
    initMissionSelector();
    refreshFilters();
    render();
  }
}

init();
