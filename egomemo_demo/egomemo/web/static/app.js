/**
 * EgoMemo Web UI - Frontend Application
 *
 * Video playback behavior:
 * - Video auto-plays when processing starts, synced to processing time
 * - When model outputs answers/proactive → video PAUSES, overlay shows all outputs
 * - User can ask follow-up questions in the pause overlay
 * - Clicking "Continue" resumes video playback
 */

// ============================================================
// State
// ============================================================
const state = {
    ws: null,
    videoPath: null,
    videoEl: null,           // <video> element reference
    videoDuration: 0,        // total video duration in seconds
    _playTimer: null,        // timer for auto-pause after short playback
    scheduledQuestions: [],
    questions: {},
    events: [],
    isRunning: false,
    isPaused: false,         // true when video is paused for model output
    pauseOutputs: [],        // pending outputs to show in overlay
    currentProcessingTime: 0,
    captionWindow: 10,         // synced from UI on start
    currentWindowEnd: 0,       // video pauses here waiting for backend
    stats: { steps: 0, answers: 0, proactive: 0, active: 0 },

    // ========== Chunk 状态机 ==========
    // 每个 chunk 生命周期（前端视角）：
    //   1) PLAY:   播第 N 段视频
    //   2) 播完 → 发 frontend_ready，等后端推理本 chunk
    //   3) WAIT:  等 step_complete
    //   4) SPEAK: step_complete 到达，串行播所有 outputs 的 TTS
    //   5) TTS 全部播完 → 回到 1) 播下一段视频
    chunkIdx: 0,               // 当前正在处理/播放的 chunk 索引
    // key=chunkIdx, value=Array<output> 已经通过 answer_ready 提前到达的 outputs
    earlyAnswersByChunk: {},
    // 已经被说过的 "唯一键"，用于去重（step_complete 里的 outputs 可能和 answer_ready 的重复）
    spokenKeys: new Set(),
};

const $ = (sel) => document.querySelector(sel);

// ============================================================
// WebSocket
// ============================================================
function connectWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    state.ws = new WebSocket(`${proto}://${location.host}/ws`);
    state.ws.onopen = () => { updateStatusDot('connected'); };
    state.ws.onclose = () => {
        updateStatusDot('disconnected');
        setTimeout(connectWebSocket, 2000);
    };
    state.ws.onmessage = (evt) => {
        try { handleEvent(JSON.parse(evt.data)); }
        catch (e) { console.error('WS parse error:', e); }
    };
}

function sendWS(msg) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify(msg));
    }
}

// ============================================================
// Event Handling
// ============================================================
function handleEvent(msg) {
    switch (msg.type) {
        case 'step_complete':
            handleStepComplete(msg);
            break;

        case 'question_added':
            state.questions[msg.qid] = {
                qid: msg.qid, text: msg.text, timestamp: msg.timestamp,
                status: 'pending', follow_up_parent: msg.follow_up_parent,
            };
            addChatBubble('user', msg.text, {
                tag: msg.qid,
                time: msg.timestamp,
            });
            // 视频内显示问题气泡
            showQuestionBubble(msg.text, msg.timestamp);
            break;

        case 'question_timeout':
            if (state.questions[msg.qid]) state.questions[msg.qid].status = 'timed_out';
            addEventCard('timeout', msg.time, `Question ${msg.qid} timed out: ${msg.text}`);
            addChatBubble('assistant', `Question ${msg.qid} timed out.`, {
                tag: 'Timeout', tagClass: 'proactive', time: msg.time,
            });
            break;

        case 'processing_started':
            state.isRunning = true;
            state.captionWindow = parseInt($('#p-caption-window').value) || 10;
            state.currentWindowEnd = state.captionWindow;
            state.chunkIdx = 0;
            state.earlyAnswersByChunk = {};
            state.spokenKeys = new Set();
            updateStatusDot('running');
            setARHudStatus('running');
            showScanOverlay();
            addEventCard('caption', 0, `Processing started: ${msg.video_path}`);
            // chunk 0：没有"上一 chunk 的 TTS"要等，立刻 frontend_ready 让后端
            // 在 mem_ready[0] 就绪后开始推理；同时前端并行开始播放 chunk 0 视频。
            sendWS({ type: 'frontend_ready' });
            playWindowSegment(0);
            break;

        case 'processing_complete':
            state.isRunning = false;
            updateStatusDot('completed');
            setARHudStatus('completed');
            hideScanOverlay();
            state.stats = {
                steps: msg.total_steps || state.stats.steps,
                answers: msg.total_answers || state.stats.answers,
                proactive: msg.total_proactive || state.stats.proactive,
                active: 0,
            };
            updateStats();
            addEventCard('caption', 0, 'Processing complete!');
            if (state.videoEl) state.videoEl.pause();
            $('#btn-start').disabled = false;
            break;

        case 'pipeline_stage':
            handlePipelineStage(msg);
            break;

        case 'answer_ready':
            // 提前到达的答案：先缓存到本 chunk 的 buffer，等 step_complete 里一起串行播 TTS，
            // 保证"视频播完 → 推理完 → TTS 一口气播完 → 下一段视频"的严格顺序。
            bufferEarlyAnswer(msg);
            break;

        case 'error':
            addEventCard('timeout', 0, `Error: ${msg.message}`);
            break;
    }
}

// ============================================================
// Video Sync & Pause/Resume
// ============================================================
/**
 * Play video for a window segment using FPS-based frame stepping.
 * Video stays paused; we manually advance currentTime at the chosen FPS.
 * This gives precise control over playback speed.
 */
let _fpsTimer = null;
let _onWindowPlaybackDone = null;  // 当前窗口视频播完的回调

function playWindowSegment(startTime, onDone) {
    if (!state.videoEl) { if (onDone) onDone(); return; }
    // 停止之前的定时器
    if (_fpsTimer) { clearInterval(_fpsTimer); _fpsTimer = null; }
    _onWindowPlaybackDone = onDone || null;

    state.videoEl.muted = true;
    state.videoEl.pause();  // 始终暂停，用定时器手动推进
    state.videoEl.currentTime = startTime;

    const fps = parseInt($('#p-playback-fps')?.value) || 15;
    const speed = parseFloat($('#p-playback-speed')?.value) || 1;
    const frameInterval = 1000 / fps;  // 每帧间隔（毫秒）
    const timeStep = speed / fps;      // 每帧推进的视频时间（秒）

    _fpsTimer = setInterval(() => {
        if (!state.videoEl || state.isPaused) {
            clearInterval(_fpsTimer);
            _fpsTimer = null;
            return;
        }
        const nextTime = state.videoEl.currentTime + timeStep;
        if (nextTime >= state.currentWindowEnd - 0.05) {
            state.videoEl.currentTime = state.currentWindowEnd;
            clearInterval(_fpsTimer);
            _fpsTimer = null;
            // 触发 timeupdate 更新 HUD
            state.videoEl.dispatchEvent(new Event('timeupdate'));
            // 视频帧播完，触发回调
            if (_onWindowPlaybackDone) {
                const cb = _onWindowPlaybackDone;
                _onWindowPlaybackDone = null;
                cb();
            }
            return; // 停在窗口末尾，等 step_complete
        }
        state.videoEl.currentTime = nextTime;
    }, frameInterval);
}

/**
 * Called from step_complete handler. Backend finished processing this window.
 * If no outputs: advance to next window immediately.
 * If has outputs: video stays paused, show overlay + avatar speaks.
 */
function syncVideoToTime(seconds) {
    if (_fpsTimer) { clearInterval(_fpsTimer); _fpsTimer = null; }
    if (state.videoEl && !state.videoEl.paused) {
        state.videoEl.pause();
    }
}

function pauseVideoWithOutputs(outputs, time) {
    state.isPaused = true;
    if (_fpsTimer) { clearInterval(_fpsTimer); _fpsTimer = null; }
    if (state.videoEl) state.videoEl.pause();

    // Dialogue bubbles + sequential TTS, then auto-continue
    speakOutputsSequentially(outputs, () => {
        setTimeout(() => {
            hideBubbles();
            resumeVideo();
        }, 2000);
    });
}

/**
 * Show answer bubble (left, next to avatar mouth).
 */
function showAnswerBubble(text, tagType, qid) {
    const bubble = $('#video-answer-bubble');
    if (!bubble) return;
    const tagClass = tagType === 'proactive' ? 'proactive' : 'answer';
    const tagLabel = tagType === 'proactive' ? 'Proactive' : (qid || 'Answer');
    bubble.innerHTML = `<div class="bubble-tag ${tagClass}">${tagLabel}</div><div>${escapeHtml(text)}</div>`;
    bubble.classList.add('active');
}

/**
 * Show question bubble (right side).
 */
function showQuestionBubble(text, qid) {
    const bubble = $('#video-question-bubble');
    if (!bubble) return;
    bubble.innerHTML = `<div class="bubble-tag question">${qid || 'Q'}</div><div>${escapeHtml(text)}</div>`;
    bubble.classList.add('active');
}

function hideBubbles() {
    const a = $('#video-answer-bubble');
    const q = $('#video-question-bubble');
    if (a) { a.classList.remove('active'); a.innerHTML = ''; }
    if (q) { q.classList.remove('active'); q.innerHTML = ''; }
}

/**
 * Speak outputs one after another with subtitle shown for each.
 */
/**
 * 根据文本内容推断情感 → 返回表情名
 */
function _detectSentiment(text) {
    const t = (text || '').toLowerCase();
    // 负面情绪（从强到弱）
    if (/hate|angry|furious|rage|fuck|damn|shit|kill|die|stupid|idiot|worst|terrible|horrible/.test(t)) return 'angry';
    if (/sad|cry|miss|lonely|depressed|upset|disappointed|heartbr|sorry|apologize|regret/.test(t)) return 'sad';
    if (/careful|warning|danger|watch out|stop|don't|caution|risk|fire|hot|sharp|hurt/.test(t)) return 'warning';
    if (/worry|concern|afraid|scared|nervous|anxious|problem|trouble|wrong|bad|fail/.test(t)) return 'concerned';
    // 正面情绪
    if (/haha|lol|funny|joke|hilarious|amazing|awesome|fantastic|wonderful|love|great|best|excellent|perfect/.test(t)) return 'excited';
    if (/happy|glad|good|nice|thank|thanks|cool|sweet|cute|beautiful|pretty|smart|clever/.test(t)) return 'happy';
    if (/wow|oh|whoa|really|no way|incredible|unbelievable|surprise/.test(t)) return 'surprised';
    if (/think|hmm|consider|maybe|perhaps|wonder|curious|what if|how about/.test(t)) return 'thinking';
    if (/blush|embarrass|shy|flatter|compliment/.test(t)) return 'shy';
    // 问句倾向 thinking
    if (/\?$/.test(t.trim())) return 'thinking';
    return 'happy';  // 默认说话时微笑
}

/**
 * 根据输出内容推断表情（用于 pipeline 输出）
 */
function _inferExpression(output) {
    if (typeof AvatarManager === 'undefined' || !AvatarManager.setExpression) return;
    const text = output.answer || output.content || '';

    if (output.type === 'proactive') {
        const sentiment = _detectSentiment(text);
        AvatarManager.setExpression(sentiment === 'happy' ? 'concerned' : sentiment);
    } else {
        AvatarManager.setExpression(_detectSentiment(text));
    }
}

function speakOutputsSequentially(outputs, onAllDone) {
    let index = 0;

    function speakNext() {
        if (index >= outputs.length) {
            // 全部说完，回到 neutral
            if (typeof AvatarManager !== 'undefined' && AvatarManager.setExpression) {
                AvatarManager.setExpression('neutral');
            }
            if (onAllDone) onAllDone();
            return;
        }
        const output = outputs[index];
        const text = output.type === 'answer' ? output.answer : output.content;
        const tagType = output.type === 'answer' ? 'answer' : 'proactive';
        const qid = output.qid || null;
        const audioUrl = output.tts_audio_url;
        index++;

        // 设置表情
        _inferExpression(output);

        // Show question bubble on the right (if answering a known question)
        if (output.type === 'answer' && qid && state.questions[qid]) {
            showQuestionBubble(state.questions[qid].text, state.questions[qid].timestamp);
        } else {
            const qb = $('#video-question-bubble');
            if (qb) { qb.classList.remove('active'); qb.innerHTML = ''; }
        }

        // Show answer bubble on the left (next to avatar)
        showAnswerBubble(text, tagType, displayTime(output.ref_timestamp, output.time));

        if (typeof AvatarManager !== 'undefined' && audioUrl) {
            AvatarManager.speak(audioUrl);
            const waitDone = () => {
                if (AvatarManager.isSpeaking) { setTimeout(waitDone, 200); }
                else { hideBubbles(); setTimeout(speakNext, 400); }
            };
            setTimeout(waitDone, 500);
        } else if (typeof AvatarManager !== 'undefined') {
            AvatarManager.speakText(text);
            const waitDone = () => {
                if (AvatarManager.isSpeaking) { setTimeout(waitDone, 200); }
                else { hideBubbles(); setTimeout(speakNext, 400); }
            };
            setTimeout(waitDone, 500);
        } else {
            setTimeout(() => { hideBubbles(); setTimeout(speakNext, 400); }, 3000);
        }
    }
    speakNext();
}

function resumeVideo() {
    state.isPaused = false;
    advanceToNextWindow();
}

function advanceToNextWindow(onDone) {
    state.currentWindowEnd += state.captionWindow;
    playWindowSegment(state.currentWindowEnd - state.captionWindow, onDone);
    showScanOverlay();
}


function submitPauseQuestion() {
    const text = $('#pause-q-text').value.trim();
    if (!text) return;

    const recurring = $('#pause-q-recurring').checked;
    sendWS({
        type: 'ask_question',
        text,
        timestamp: state.currentProcessingTime,
        recurring,
    });
    $('#pause-q-text').value = '';
    $('#pause-q-text').placeholder = 'Question submitted!';
    setTimeout(() => { $('#pause-q-text').placeholder = 'Type your question...'; }, 1500);
}

// ============================================================
// Event Log Rendering
// ============================================================
function addEventCard(type, time, content, evidence, timeSpan, qid, eventId) {
    const log = $('#event-log-content');
    const card = document.createElement('div');
    card.className = `event-card ${type}`;
    const timeStr = typeof time === 'number' ? formatTime(time) : time;
    let labelText = type.charAt(0).toUpperCase() + type.slice(1);
    if (qid) labelText += ` ${qid}`;

    let html = `
        <div class="event-time">${timeStr}${timeSpan ? ' | ' + timeSpan : ''}</div>
        <div class="event-label ${type}">${labelText}</div>
        <div class="event-content">${escapeHtml(content || '')}</div>
    `;
    if (evidence) {
        html += `<div class="event-evidence">Evidence: ${escapeHtml(evidence)}</div>`;
    }
    if (type === 'proactive' && eventId) {
        html += `
            <div class="follow-up-row" id="followup-${eventId}">
                <input type="text" placeholder="Follow up..."
                    onkeydown="if(event.key==='Enter')sendFollowUp('${eventId}', this)" />
                <button class="btn btn-small btn-secondary"
                    onclick="sendFollowUp('${eventId}', this.previousElementSibling)">Reply</button>
            </div>
        `;
    }
    card.innerHTML = html;
    log.appendChild(card);
    log.scrollTop = log.scrollHeight;
    state.events.push({ type, time, content });
}

// ============================================================
// Chat Bubble Rendering (right panel)
// ============================================================
const TRACE_ICONS = {
    think: '\u{1F4A1}',       // 💡
    search: '\u{1F50D}',      // 🔍
    observation: '\u{1F441}', // 👁
    respond: '\u{1F4AC}',     // 💬
};

function _renderReasoningTrace(trace) {
    if (!trace || !trace.length) return '';
    const steps = trace.map((t, i) => {
        const icon = TRACE_ICONS[t.step] || '●';
        const delay = i * 0.3;  // 每步延迟 0.3s 逐步出现
        return `
            <div class="reasoning-step ${t.step}" style="animation-delay:${delay}s;">
                <div class="reasoning-step-icon">${icon}</div>
                <div class="reasoning-step-body">
                    <div class="reasoning-step-label">{${t.step}}</div>
                    <div class="reasoning-step-content">${escapeHtml(t.content)}</div>
                </div>
            </div>
        `;
    }).join('');
    return `<div class="reasoning-trace">${steps}</div>`;
}

function addChatBubble(role, content, options = {}) {
    // role: 'assistant' or 'user'
    // options: { tag, tagClass, time, isProactive, qid, reasoning_trace }
    const container = $('#chat-messages');

    const row = document.createElement('div');
    row.className = `chat-bubble-row ${role}${options.isProactive ? ' proactive' : ''}`;

    const avatarIcon = role === 'assistant' ? '&#x1F916;' : '&#x1F464;';
    const roleLabel = role === 'assistant' ? 'Assistant' : 'User';

    let tagHtml = '';
    if (options.tag) {
        tagHtml = `<span class="chat-bubble-tag ${options.tagClass || ''}">${escapeHtml(options.tag)}</span>`;
    }

    let timeHtml = '';
    if (options.time !== undefined && options.time !== null) {
        const t = (typeof options.time === 'number') ? formatTime(options.time) : options.time;
        timeHtml = `<div class="chat-bubble-time">${t}</div>`;
    }

    // 推理过程动画
    let traceHtml = '';
    if (options.reasoning_trace && options.reasoning_trace.length > 0) {
        traceHtml = _renderReasoningTrace(options.reasoning_trace);
    }

    row.innerHTML = `
        <div class="chat-avatar">${avatarIcon}</div>
        <div class="chat-bubble-body">
            <div class="chat-bubble-role">${roleLabel}</div>
            ${tagHtml}
            ${traceHtml}
            <div class="chat-bubble">${escapeHtml(content)}</div>
            ${timeHtml}
        </div>
    `;

    container.appendChild(row);
    container.scrollTop = container.scrollHeight;
}

function updateQuestionStatus(qid, status, answer) {
    if (state.questions[qid]) {
        state.questions[qid].status = status;
        if (answer) state.questions[qid].answer = answer;
    }
}

function sendChatQuestion() {
    const text = $('#chat-input').value.trim();
    if (!text) return;
    const recurring = $('#chat-recurring').checked;

    // 没有 pipeline 在跑就直接提示，不然问题会被后端静默丢弃
    if (!state.isRunning) {
        alert('Please click Start first — the pipeline is not running yet.');
        return;
    }

    // Show user bubble immediately
    addChatBubble('user', text, { time: state.currentProcessingTime });

    // Send via WebSocket
    sendWS({
        type: 'ask_question',
        text,
        timestamp: state.currentProcessingTime,
        recurring,
    });
    $('#chat-input').value = '';
    $('#chat-recurring').checked = false;

    // 同时用年轻男声把问题朗读一遍（不阻塞，失败也不影响提交）
    _speakQuestionAsMale(text);
}

// 用年轻男声朗读用户刚提交的问题（独立音频元素，不走 Live2D 对嘴）
// 用一个全局引用 + 队列，解决以下问题：
//   1) Audio 对象被 GC 提前回收导致播一半没声
//   2) 连续点 Send 时多个 TTS 请求并发，互相抢占音频
//   3) 浏览器 autoplay policy 拦截 —— 失败时短暂延迟后重试
window._questionTtsQueue = window._questionTtsQueue || [];
window._questionTtsPlaying = false;

async function _speakQuestionAsMale(text) {
    const voice = 'en-US-AndrewNeural';
    try {
        const resp = await fetch(
            '/api/test_avatar?text=' + encodeURIComponent(text) +
            '&voice=' + encodeURIComponent(voice)
        );
        if (!resp.ok) {
            console.warn('[question TTS] fetch not ok:', resp.status);
            return;
        }
        const data = await resp.json();
        if (!data || !data.tts_audio_url) {
            console.warn('[question TTS] no audio url in response:', data);
            return;
        }
        window._questionTtsQueue.push(data.tts_audio_url);
        _drainQuestionTtsQueue();
    } catch (e) {
        console.warn('[question TTS] error:', e);
    }
}

function _drainQuestionTtsQueue() {
    if (window._questionTtsPlaying) return;
    const url = window._questionTtsQueue.shift();
    if (!url) return;
    window._questionTtsPlaying = true;

    const audio = new Audio(url);
    // 挂到 window 防止被 GC
    window._questionTtsAudio = audio;

    const cleanup = () => {
        window._questionTtsPlaying = false;
        // 播完后马上尝试下一条
        _drainQuestionTtsQueue();
    };
    audio.onended = cleanup;
    audio.onerror = (e) => { console.warn('[question TTS] audio error:', e); cleanup(); };

    const tryPlay = (retried) => {
        audio.play().catch(err => {
            console.warn('[question TTS] play rejected:', err);
            if (!retried) {
                // autoplay 被拦可能是因为没有用户交互上下文 —— 延迟一点再试
                setTimeout(() => tryPlay(true), 300);
            } else {
                cleanup();
            }
        });
    };
    tryPlay(false);
}

// ============================================================
// Video Upload
// ============================================================
function initDropZone() {
    const zone = $('#drop-zone');
    const fileInput = $('#file-input');
    zone.addEventListener('click', () => fileInput.click());
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) uploadVideo(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) uploadVideo(e.target.files[0]);
    });
}

async function uploadVideo(file) {
    const form = new FormData();
    form.append('file', file);
    const dropZone = $('#drop-zone');
    const sizeMB = (file.size / 1024 / 1024).toFixed(1);
    dropZone.innerHTML = `<p>Uploading 0% (0 / ${sizeMB} MB)</p>`;

    try {
        const data = await new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/upload_video');
            xhr.timeout = 10 * 60 * 1000; // 10分钟超时
            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const pct = ((e.loaded / e.total) * 100).toFixed(1);
                    const doneMB = (e.loaded / 1024 / 1024).toFixed(1);
                    dropZone.innerHTML = `<p>Uploading ${pct}% (${doneMB} / ${sizeMB} MB)</p>`;
                }
            };
            xhr.upload.onloadend = () => {
                dropZone.innerHTML = `<p>Upload done, waiting for server to finish writing...</p>`;
            };
            xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try { resolve(JSON.parse(xhr.responseText)); }
                    catch (err) { reject(new Error('Bad JSON: ' + xhr.responseText)); }
                } else {
                    reject(new Error(`HTTP ${xhr.status}: ${xhr.responseText || xhr.statusText}`));
                }
            };
            xhr.onerror = () => reject(new Error('Network error (server may have crashed)'));
            xhr.ontimeout = () => reject(new Error('Upload timed out after 10 min'));
            xhr.onabort = () => reject(new Error('Upload aborted'));
            xhr.send(form);
        });

        if (data.video_path) {
            state.videoPath = data.video_path;
            showVideo(file);
        } else {
            dropZone.innerHTML = `<p style="color:#f66">Upload failed: ${JSON.stringify(data)}</p><p style="font-size:12px">Click to retry</p>`;
            dropZone.onclick = () => $('#file-input').click();
        }
    } catch (e) {
        dropZone.innerHTML = `<p style="color:#f66">Upload error: ${e.message}</p><p style="font-size:12px">Click to retry</p>`;
        dropZone.onclick = () => $('#file-input').click();
    }
}

function showVideo(file) {
    showVideoFromUrl(URL.createObjectURL(file));
}

function showVideoFromUrl(url) {
    const container = $('#video-container');

    // Remove drop-zone but keep avatar overlay and subtitle
    const dropZone = $('#drop-zone');
    if (dropZone) dropZone.remove();

    // Remove old video if any
    const oldVideo = $('#main-video');
    if (oldVideo) oldVideo.remove();

    // Insert video element at the beginning of container (behind overlays)
    const video = document.createElement('video');
    video.id = 'main-video';
    video.src = url;
    video.muted = true;
    video.playsInline = true;
    container.insertBefore(video, container.firstChild);

    state.videoEl = video;
    state.videoEl.addEventListener('loadedmetadata', () => {
        state.videoDuration = state.videoEl.duration;
        console.log('Video loaded, duration:', state.videoDuration);
    });

    // 实时更新 HUD 时间和帧数
    state.videoEl.addEventListener('timeupdate', () => {
        const timeEl = document.getElementById('ar-hud-time');
        const frameEl = document.getElementById('ar-hud-frame');
        if (timeEl) timeEl.textContent = formatTime(state.videoEl.currentTime);
        if (frameEl) frameEl.textContent = 'Frame ' + Math.floor(state.videoEl.currentTime * 30);
    });
}

// 预设视频：从服务器加载已有视频（绕开慢速上传链路）
async function loadPresetVideo(ev) {
    if (ev) ev.stopPropagation();  // 防止点击冒泡到 drop-zone 触发文件选择器
    const dropZone = $('#drop-zone');
    try {
        const resp = await fetch('/api/preset_video');
        const data = await resp.json();
        if (!data.available) {
            if (dropZone) dropZone.innerHTML = `<p style="color:#f66">No preset video configured on server.</p>`;
            return;
        }
        state.videoPath = data.path;            // 给后端 pipeline 用的绝对路径
        showVideoFromUrl(data.stream_url);      // 给 <video> 用的 HTTP stream URL
        console.log('Preset video loaded:', data.path);
    } catch (e) {
        if (dropZone) dropZone.innerHTML = `<p style="color:#f66">Load preset failed: ${e.message}</p>`;
    }
}

// 页面加载后查一下有没有预设视频，有就显示按钮
async function initPresetButton() {
    try {
        const resp = await fetch('/api/preset_video');
        const data = await resp.json();
        const btn = $('#btn-load-preset');
        if (btn && data.available) {
            btn.textContent = data.label || 'Load Preset Video';
            btn.style.display = 'inline-block';
        }
    } catch (e) {
        console.warn('preset video check failed:', e);
    }
}
document.addEventListener('DOMContentLoaded', initPresetButton);

// ============================================================
// Question Scheduling
// ============================================================
function addScheduledQuestion() {
    const text = $('#q-text').value.trim();
    const time = parseFloat($('#q-time').value) || 0;
    const recurring = $('#q-recurring').checked;
    if (!text) return;
    state.scheduledQuestions.push({ text, timestamp: time, recurring });
    $('#q-text').value = '';
    $('#q-recurring').checked = false;
    renderScheduledQuestions();
}

function removeScheduledQuestion(idx) {
    state.scheduledQuestions.splice(idx, 1);
    renderScheduledQuestions();
}

function renderScheduledQuestions() {
    const container = $('#scheduled-questions');
    if (!state.scheduledQuestions.length) {
        container.innerHTML = '<p style="font-size:12px;color:var(--text-secondary);padding:4px;">No questions scheduled yet.</p>';
        return;
    }
    container.innerHTML = state.scheduledQuestions.map((q, i) => `
        <div class="scheduled-q">
            <span class="q-time">${q.timestamp}s</span>
            <span class="q-text">${escapeHtml(q.text)}${q.recurring ? ' <em>(recurring)</em>' : ''}</span>
            <span class="q-remove" onclick="removeScheduledQuestion(${i})">x</span>
        </div>
    `).join('');
}

// ============================================================
// Start Processing
// ============================================================
async function startProcessing() {
    if (!state.videoPath) { alert('Please upload a video first.'); return; }

    // Collect pipeline parameters from UI
    const params = {
        step_interval: parseFloat($('#p-step-interval').value) || 2,
        caption_window: parseInt($('#p-caption-window').value) || 10,
        proactive_cooldown: parseFloat($('#p-proactive-cooldown').value) || 30,
        clear_cache: $('#p-clear-cache').checked,
        datasets_type: ($('#p-datasets-type') && $('#p-datasets-type').value) || 'holoassist',
        hydration_reminder_enabled: ($('#p-hydration-reminder') && $('#p-hydration-reminder').checked) || false,
        circuit_breaker_scene_enabled: ($('#p-circuit-breaker-scene') && $('#p-circuit-breaker-scene').checked) || false,
        egg_recipe_guidance_enabled: ($('#p-egg-recipe') && $('#p-egg-recipe').checked) || false,
    };

    const form = new FormData();
    form.append('video_path', state.videoPath);
    form.append('questions', JSON.stringify(state.scheduledQuestions));
    form.append('params', JSON.stringify(params));

    try {
        const resp = await fetch('/api/start_processing', { method: 'POST', body: form });
        const data = await resp.json();
        if (data.status === 'started') {
            state.isRunning = true;
            updateStatusDot('running');
            $('#btn-start').disabled = true;
        } else {
            alert('Start failed: ' + JSON.stringify(data));
        }
    } catch (e) {
        alert('Start error: ' + e.message);
    }
}

// ============================================================
// Follow-up
// ============================================================
function sendFollowUp(eventId, inputEl) {
    const text = inputEl.value.trim();
    if (!text) return;
    sendWS({ type: 'follow_up', text, proactive_event_id: eventId, timestamp: 0 });
    inputEl.value = '';
    inputEl.placeholder = 'Sent!';
    setTimeout(() => { inputEl.placeholder = 'Follow up...'; }, 1500);
}

// ============================================================
// UI Helpers
// ============================================================
function updateStatusDot(status) {
    const dot = $('#status-dot');
    const label = $('#status-label');
    dot.className = 'status-dot';
    if (status === 'running') { dot.classList.add('running'); label.textContent = 'Processing...'; }
    else if (status === 'completed') { dot.classList.add('completed'); label.textContent = 'Completed'; }
    else if (status === 'connected') { label.textContent = 'Connected'; }
    else { label.textContent = 'Disconnected'; }
}

function updateStats() {
    _bumpStat('stat-steps', state.stats.steps);
    _bumpStat('stat-answers', state.stats.answers);
    _bumpStat('stat-proactive', state.stats.proactive);
}

function _bumpStat(id, newVal) {
    const el = $('#' + id);
    if (!el) return;
    if (el.textContent !== String(newVal)) {
        el.textContent = newVal;
        el.classList.add('stat-updated');
        setTimeout(() => el.classList.remove('stat-updated'), 400);
    }
}

// ============================================================
// Reasoning Process Panel
// ============================================================
const STAGE_ICONS = {
    think: '&#x1F4AD;',      // 💭
    search: '&#x1F50D;',     // 🔍
    observation: '&#x1F4CB;', // 📋
    respond: '&#x1F4AC;',    // 💬
};

const STAGE_LABELS = {
    think: 'Think',
    search: 'Search',
    observation: 'Observation',
    respond: 'Respond',
};

function addReasoningTrace(qid, time, trace) {
    const container = document.getElementById('reasoning-content');
    if (!container) return;

    // 移除 placeholder
    const placeholder = container.querySelector('.reasoning-placeholder');
    if (placeholder) placeholder.remove();

    const block = document.createElement('div');
    block.className = 'reasoning-block';

    // Header: question ID + time
    const header = document.createElement('div');
    header.className = 'reasoning-block-header';
    header.textContent = `${qid ? '[' + qid + ']' : '[Proactive]'} @ ${formatTime(time)}`;
    block.appendChild(header);

    // Render each stage with animation delay
    trace.forEach((step, idx) => {
        if (idx > 0) {
            const connector = document.createElement('div');
            connector.className = 'reasoning-connector';
            block.appendChild(connector);
        }

        const stage = document.createElement('div');
        stage.className = `reasoning-stage stage-${step.stage}`;
        stage.style.animationDelay = `${idx * 0.3}s`;
        stage.style.opacity = '0';
        stage.style.animation = `reasonFadeIn 0.4s ease-out ${idx * 0.3}s forwards`;

        stage.innerHTML = `
            <div class="reasoning-stage-icon">${STAGE_ICONS[step.stage] || '?'}</div>
            <div class="reasoning-stage-body">
                <div class="reasoning-stage-label">${STAGE_LABELS[step.stage] || step.stage}</div>
                <div class="reasoning-stage-text">${escapeHtml(step.content || '')}</div>
            </div>
        `;
        block.appendChild(stage);
    });

    container.appendChild(block);
    // 自动滚到底部
    container.scrollTop = container.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================
// Chunk 状态机核心：bufferEarlyAnswer / handleStepComplete
// ============================================================

let _expressionResetTimer = null;

// answer_ready 到达时调用：只缓存，附带更新面板/气泡，不触发 TTS
// 注意：answer_ready 里 proactive 的 msg.qid 其实就是 event_id（见后端 pipeline_v2_patch.py）
function bufferEarlyAnswer(msg) {
    const chunkIdx = (typeof msg.step === 'number') ? msg.step : state.chunkIdx;
    if (!state.earlyAnswersByChunk[chunkIdx]) state.earlyAnswersByChunk[chunkIdx] = [];

    const isProactive = !!msg.is_proactive;
    const output = {
        type: isProactive ? 'proactive' : 'answer',
        qid: isProactive ? null : (msg.qid || null),
        event_id: isProactive ? (msg.qid || null) : null,   // proactive 的 qid == event_id
        answer: msg.answer,
        content: msg.answer,
        time: msg.time,
        ref_timestamp: msg.ref_timestamp || null,
        tts_audio_url: msg.tts_audio_url || null,
    };
    state.earlyAnswersByChunk[chunkIdx].push(output);

    _appendRespondToPanel(msg.qid || 'proactive', msg.time, msg.answer, msg.ref_timestamp);
}

// step_complete 到达时调用：合并 early + 新 outputs，串行播 TTS，
// TTS 全播完后播下一段视频，视频播完再 notify frontend_ready
function handleStepComplete(msg) {
    state.stats.steps = (msg.step || 0) + 1;
    state.stats.active = msg.active_questions || 0;
    state.currentProcessingTime = msg.time || 0;

    syncVideoToTime(msg.time);

    if (msg.caption) {
        addEventCard('caption', msg.time, msg.caption, null, msg.time_span);
    }

    const chunkIdx = (typeof msg.step === 'number') ? msg.step : state.chunkIdx;

    // 合并 early + step_complete 里的 outputs，按 (type, qid|event_id, time) 去重。
    // early 先插入；如果 step 里有同 key 的，不重复插入但补上 tts_audio_url（早版本可能 null）。
    const early = state.earlyAnswersByChunk[chunkIdx] || [];
    const fromStep = msg.outputs || [];
    const keyOf = (o) => {
        const id = o.type === 'answer' ? (o.qid || '') : (o.event_id || '');
        return `${o.type}:${id}:${o.time}`;
    };
    const byKey = new Map();
    const merged = [];
    for (const o of early) {
        const k = keyOf(o);
        if (!byKey.has(k)) { byKey.set(k, o); merged.push(o); }
    }
    for (const o of fromStep) {
        const k = keyOf(o);
        const existed = byKey.get(k);
        if (!existed) { byKey.set(k, o); merged.push(o); }
        else if (!existed.tts_audio_url && o.tts_audio_url) {
            existed.tts_audio_url = o.tts_audio_url;
        }
    }
    delete state.earlyAnswersByChunk[chunkIdx];

    // 侧栏 / chat bubble / stats（原来的视觉反馈保留）
    merged.forEach(output => {
        const displayT = displayTime(output.ref_timestamp, output.time);
        if (output.type === 'answer') {
            state.stats.answers++;
            addEventCard('answer', displayT, output.answer,
                output.evidence, null, output.qid);
            updateQuestionStatus(output.qid, 'answered', output.answer);
            addChatBubble('assistant', output.answer, {
                tag: `Answer ${output.qid}`, tagClass: 'answer', time: displayT,
            });
        } else if (output.type === 'proactive') {
            state.stats.proactive++;
            addEventCard('proactive', displayT, output.content,
                output.evidence, null, null, output.event_id);
            addChatBubble('assistant', output.content, {
                tag: 'Proactive', tagClass: 'proactive',
                time: displayT, isProactive: true,
            });
        }
    });
    if ((msg.actions || []).some(a => a.includes('MEM_READ'))) {
        addEventCard('mem-read', msg.time, 'Memory retrieval requested...');
    }
    updateStats();
    updateARHud();
    hideScanOverlay();

    // 严格顺序：本 chunk 所有 TTS 串行播完 → 同一时刻触发：
    //   - 前端开始播 chunk N+1 的视频段
    //   - 发 frontend_ready 给后端（让后端在 mem_ready[N+1] 也就绪时开始推理 N+1）
    // 这样 chunk N+1 的"前端播放"和"后端推理"完全并行、同步起跑。
    speakOutputsSerial(merged, () => {
        state.chunkIdx = chunkIdx + 1;
        advanceToNextWindow();
        sendWS({ type: 'frontend_ready' });
    });
}

// 严格串行播 outputs 的 TTS。AvatarManager.speak/speakText 返回 Promise，
// 用 await 保证"上一条 TTS 彻底播完后才开始下一条"，彻底消除 race。
async function speakOutputsSerial(outputs, onAllDone) {
    try {
        const list = outputs || [];
        for (let idx = 0; idx < list.length; idx++) {
            const o = list[idx];
            const isLast = (idx === list.length - 1);
            const text = o.type === 'answer' ? o.answer : o.content;
            const tagType = o.type === 'answer' ? 'answer' : 'proactive';

            // 表情
            if (typeof AvatarManager !== 'undefined' && AvatarManager.setExpression) {
                AvatarManager.setExpression(o.type === 'proactive' ? 'concerned' : 'happy');
            }

            // 气泡：answer 附带问题气泡（右侧），所有输出显示答案气泡（左侧）。
            // 在 show 之前不 hide，直接替换内容，保证切换无闪烁。
            if (o.type === 'answer' && o.qid && state.questions[o.qid]) {
                showQuestionBubble(state.questions[o.qid].text, state.questions[o.qid].timestamp);
            } else {
                const qb = $('#video-question-bubble');
                if (qb) { qb.classList.remove('active'); qb.innerHTML = ''; }
            }
            showAnswerBubble(text, tagType, displayTime(o.ref_timestamp, o.time));

            // 播 TTS，气泡显示时长 == TTS 播放时长（await 精确对齐）。
            if (o.tts_audio_url && AvatarManager && AvatarManager.speak) {
                await AvatarManager.speak(o.tts_audio_url);
            } else {
                // 后端未生成音频时的静默展示时长（按字数估算）
                const readMs = Math.max(1200, Math.min(8000, (text || '').length * 70));
                console.warn('[TTS] no audio_url, silent pause', readMs, 'ms:', text);
                await new Promise(r => setTimeout(r, readMs));
            }

            // TTS 刚结束：最后一条立刻 hide；否则气泡继续挂 1.5s 让用户读完，
            // 然后直接被下一条的 showAnswerBubble 覆盖（零闪烁切换）。
            if (isLast) {
                hideBubbles();
            } else {
                await new Promise(r => setTimeout(r, 1500));
            }
        }
    } finally {
        _expressionResetTimer = setTimeout(() => {
            if (typeof AvatarManager !== 'undefined' && AvatarManager.setExpression) {
                AvatarManager.setExpression('neutral');
            }
        }, 1500);
        hideBubbles();
        if (onAllDone) onAllDone();
    }
}

// 兼容旧调用点（如果别的地方还在用 handleAnswerReady）
function handleAnswerReady(msg) { bufferEarlyAnswer(msg); }

// --- pipeline_stage 事件：只处理推理阶段，忽略记忆构造 ---
function handlePipelineStage(msg) {
    savePipelineLog(msg);

    // 忽略记忆构造阶段
    if (msg.phase === 'memory') return;

    // 推理阶段：直接追加到面板（不用队列延迟）
    if (msg.phase === 'reasoning') {
        const container = document.getElementById('reasoning-content');
        if (!container) return;

        const placeholder = container.querySelector('.reasoning-placeholder');
        if (placeholder) placeholder.remove();

        const stage = msg.stage;
        const qid = msg.qid || 'proactive';

        if (stage === 'think_start') {
            // 新问题开始：追加一个新的推理块
            const isProactive = qid.startsWith('proactive');
            const block = _createStageBlock(container,
                isProactive ? `[Proactive] @ ${formatTime(msg.time)}` : `[${qid}] @ ${formatTime(msg.time)}`);
            block.dataset.qid = qid;
            if (isProactive) block.classList.add('reasoning-block-proactive');

            if (msg.question_text) {
                _addStageNode(block, 'question', '❓', `Question @ ${formatTime(msg.time)}`, msg.question_text, 'info');
                _addConnector(block);
            }
            _addStageNode(block, 'think', '💭', 'Think', `Analyzing...`, 'active');
        }
        else if (stage === 'think_done') {
            // 模型思考完成：显示原始输出（高亮标签）
            const block = _findLastBlock(container, qid);
            if (block) {
                const rawText = msg.raw_response || '';
                const highlighted = _highlightTags(rawText);
                _updateLastNodeHtml(block, 'think-done', '💭', 'Think', highlighted, 'done');
            }
        }
        else if (stage === 'silent') {
            const block = _findLastBlock(container, qid);
            if (block) {
                _updateLastNode(block, 'silent', '😶', 'Silent', 'Nothing relevant at this moment.', 'muted');
            }
        }
        else if (stage === 'search_start') {
            const block = _findLastBlock(container, qid);
            if (block) {
                _addConnector(block);
                _addStageNode(block, 'search', '🔍', 'Search', `"${msg.search_query || ''}"`, 'active');
            }
        }
        else if (stage === 'observation') {
            const block = _findLastBlock(container, qid);
            if (block) {
                _updateLastNode(block, 'search-done', '🔍', 'Search Complete', '', 'done');
                _addConnector(block);
                _addStageNode(block, 'observation', '📋', 'Observation',
                    msg.retrieved_preview || '(no results)', 'done');
            }
        }
        else if (stage === 'search_complete_hidden') {
            // demo 模式：只把 search 节点标记为完成，不展示 observation 节点
            const block = _findLastBlock(container, qid);
            if (block) {
                _updateLastNode(block, 'search-done', '🔍', 'Search Complete', '', 'done');
            }
        }

        container.scrollTop = container.scrollHeight;
    }
}

// --- 在面板追加 respond 节点 ---
function _appendRespondToPanel(qid, time, answer, refTimestamp) {
    const container = document.getElementById('reasoning-content');
    if (!container) return;

    const placeholder = container.querySelector('.reasoning-placeholder');
    if (placeholder) placeholder.remove();

    const displayT = displayTime(refTimestamp, time);

    // 找到对应 qid 的块，或创建一个新的
    let block = _findLastBlock(container, qid);
    if (!block) {
        const isProactive = qid && qid.startsWith('proactive');
        block = _createStageBlock(container,
            isProactive ? `[Proactive] @ ${displayT}` : `[${qid || '?'}] @ ${displayT}`);
        block.dataset.qid = qid;
        if (isProactive) block.classList.add('reasoning-block-proactive');
    }

    // 更新 think 节点
    const activeNodes = block.querySelectorAll('.stage-think');
    activeNodes.forEach(n => {
        n.className = n.className.replace('stage-think', 'stage-respond');
        const spinner = n.querySelector('.stage-spinner');
        if (spinner) spinner.remove();
    });

    _addConnector(block);
    _addStageNode(block, 'respond', '💬', `Respond @ ${displayT}`, answer, 'done');

    container.scrollTop = container.scrollHeight;
}

function _findLastBlock(container, qid) {
    const blocks = container.querySelectorAll('.reasoning-block');
    for (let i = blocks.length - 1; i >= 0; i--) {
        if (blocks[i].dataset.qid === qid) return blocks[i];
    }
    return null;
}

// --- Video bubbles ---
function showAnswerBubble(text, type, time) {
    const bubble = document.getElementById('video-answer-bubble');
    if (!bubble) return;
    const tagClass = type === 'proactive' ? 'proactive' : 'answer';
    const tagLabel = type === 'proactive' ? 'PROACTIVE' : 'ANSWER';
    // time 可以是秒数（number）或已经格式化的字符串（string）
    let timeStr = '';
    if (time != null) {
        const t = (typeof time === 'number') ? formatTime(time) : time;
        timeStr = `<span class="bubble-time">${t}</span>`;
    }
    bubble.innerHTML = `<span class="bubble-tag ${tagClass}">${tagLabel}</span>${timeStr}<br>${escapeHtml(text)}`;
    bubble.classList.add('active');
    clearTimeout(bubble._hideTimer);
    bubble._hideTimer = setTimeout(() => bubble.classList.remove('active'), 6000);
}

function showQuestionBubble(text, time) {
    const bubble = document.getElementById('video-question-bubble');
    if (!bubble) return;
    const timeStr = (time != null) ? `<span class="bubble-time">${formatTime(time)}</span>` : '';
    bubble.innerHTML = `<span class="bubble-tag question">QUESTION</span>${timeStr}<br>${escapeHtml(text)}`;
    bubble.classList.add('active');
    clearTimeout(bubble._hideTimer);
    bubble._hideTimer = setTimeout(() => bubble.classList.remove('active'), 5000);
}

// --- Helper functions ---
function _createStageBlock(container, title) {
    const block = document.createElement('div');
    block.className = 'reasoning-block';
    const header = document.createElement('div');
    header.className = 'reasoning-block-header';
    header.textContent = title;
    block.appendChild(header);
    container.appendChild(block);
    return block;
}

function _addStageNode(block, id, icon, label, text, status) {
    const node = document.createElement('div');
    node.className = `reasoning-stage stage-${_stageColor(status)}`;
    node.dataset.stageId = id;
    node.innerHTML = `
        <div class="reasoning-stage-icon">${icon}</div>
        <div class="reasoning-stage-body">
            <div class="reasoning-stage-label">${label}${status === 'active' ? ' <span class="stage-spinner">⟳</span>' : ''}</div>
            <div class="reasoning-stage-text">${escapeHtml(text)}</div>
        </div>
    `;
    block.appendChild(node);
}

function _updateLastNode(block, id, icon, label, text, status) {
    const nodes = block.querySelectorAll('.reasoning-stage');
    const last = nodes[nodes.length - 1];
    if (last) {
        last.className = `reasoning-stage stage-${_stageColor(status)}`;
        last.dataset.stageId = id;
        const labelEl = last.querySelector('.reasoning-stage-label');
        const textEl = last.querySelector('.reasoning-stage-text');
        if (labelEl) labelEl.innerHTML = `${label}`;
        if (textEl && text) textEl.textContent = text;
    }
}

function _updateLastNodeHtml(block, id, icon, label, html, status) {
    const nodes = block.querySelectorAll('.reasoning-stage');
    const last = nodes[nodes.length - 1];
    if (last) {
        last.className = `reasoning-stage stage-${_stageColor(status)}`;
        last.dataset.stageId = id;
        const labelEl = last.querySelector('.reasoning-stage-label');
        const textEl = last.querySelector('.reasoning-stage-text');
        if (labelEl) labelEl.innerHTML = `${label}`;
        if (textEl) textEl.innerHTML = html;
    }
}

function _highlightTags(text) {
    // 高亮 [search]、[respond]、[silent]、[observation] 等标签
    let html = escapeHtml(text);
    html = html.replace(/\[search\]/gi, '<span class="tag-highlight tag-search">[search]</span>');
    html = html.replace(/\[respond\]/gi, '<span class="tag-highlight tag-respond">[respond]</span>');
    html = html.replace(/\[silent\]/gi, '<span class="tag-highlight tag-silent">[silent]</span>');
    html = html.replace(/\[observation\]/gi, '<span class="tag-highlight tag-observation">[observation]</span>');
    html = html.replace(/\[observe\]/gi, '<span class="tag-highlight tag-observation">[observe]</span>');
    // 截断显示
    if (html.length > 400) html = html.substring(0, 400) + '<span style="color:var(--text-secondary);">...</span>';
    return html;
}

function _addConnector(block) {
    const c = document.createElement('div');
    c.className = 'reasoning-connector';
    block.appendChild(c);
}

function _stageColor(status) {
    switch (status) {
        case 'active': return 'think';
        case 'done': return 'respond';
        case 'muted': return 'observation';
        case 'info': return 'search';
        default: return 'think';
    }
}

// --- Pipeline log saving ---
const _pipelineLogs = [];

function savePipelineLog(msg) {
    _pipelineLogs.push({ timestamp: Date.now(), ...msg });
    if (_pipelineLogs.length > 500) _pipelineLogs.splice(0, 100);
}

window.getPipelineLogs = () => JSON.stringify(_pipelineLogs, null, 2);
window.downloadPipelineLogs = () => {
    const blob = new Blob([JSON.stringify(_pipelineLogs, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pipeline_log_${Date.now()}.json`;
    a.click();
};

// ============================================================
// AR HUD + Scan Overlay
// ============================================================
function updateARHud() {
    const timeEl = document.getElementById('ar-hud-time');
    const stepEl = document.getElementById('ar-hud-step');
    const frameEl = document.getElementById('ar-hud-frame');
    if (timeEl) timeEl.textContent = formatTime(state.currentProcessingTime);
    if (stepEl) stepEl.textContent = 'Step ' + state.stats.steps;
    if (frameEl && state.videoEl) {
        frameEl.textContent = 'Frame ' + Math.floor(state.videoEl.currentTime * 30);
    }
}

function setARHudStatus(status) {
    const el = document.getElementById('ar-hud-status');
    if (!el) return;
    const dot = el.querySelector('.ar-hud-dot');
    if (status === 'running') {
        el.innerHTML = '<span class="ar-hud-dot running"></span> PROCESSING';
    } else if (status === 'completed') {
        el.innerHTML = '<span class="ar-hud-dot"></span> COMPLETE';
    } else {
        el.innerHTML = '<span class="ar-hud-dot"></span> STANDBY';
    }
}

function showScanOverlay() {
    const el = document.getElementById('ar-scan-overlay');
    if (el) el.classList.add('active');
}

function hideScanOverlay() {
    const el = document.getElementById('ar-scan-overlay');
    if (el) el.classList.remove('active');
}

function formatTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

// 用于展示 respond 的时间：优先用模型输出的精确时间戳 ref_timestamp
// （格式 "DAY1-HH:MM:SS"），否则 fallback 到 video chunk 的秒数。
function displayTime(refTimestamp, fallbackSeconds) {
    if (typeof refTimestamp === 'string' && refTimestamp) {
        // "DAY1-HH:MM:SS" -> "HH:MM:SS"
        const m = refTimestamp.match(/^DAY\d+-(\d{2}:\d{2}:\d{2})$/);
        if (m) return m[1];
        // 不是标准格式，原样返回
        return refTimestamp;
    }
    if (typeof fallbackSeconds === 'number') return formatTime(fallbackSeconds);
    return '';
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ============================================================
// Init
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    initDropZone();
    renderScheduledQuestions();
    updateStats();

    // 图片加载后动态对齐光锥起点到眼镜位置
    const boyImg = document.querySelector('.ar-boy-img');
    if (boyImg) {
        const alignCone = () => {
            const dot = document.querySelector('.ar-camera-dot');
            const stage = document.querySelector('.video-stage');
            const videoBox = document.querySelector('.video-container');
            if (!dot || !stage || !videoBox) return;
            const dotRect = dot.getBoundingClientRect();
            const stageRect = stage.getBoundingClientRect();
            const videoRect = videoBox.getBoundingClientRect();
            // 光点在 stage 中的 SVG 坐标
            const relX = ((dotRect.left + dotRect.width/2) - stageRect.left) / stageRect.width;
            const relY = ((dotRect.top + dotRect.height/2) - stageRect.top) / stageRect.height;
            const svgX = Math.round(relX * 1000);
            const svgY = Math.round(relY * 600);
            // 视频框左边缘在 stage 中的 SVG x 坐标
            const videoLeftRel = (videoRect.left - stageRect.left) / stageRect.width;
            const videoTopRel = (videoRect.top - stageRect.top) / stageRect.height;
            const videoBottomRel = (videoRect.bottom - stageRect.top) / stageRect.height;
            const vx = Math.round(videoLeftRel * 1000);
            const vy0 = Math.round(videoTopRel * 600);
            const vy1 = Math.round(videoBottomRel * 600);
            console.log(`[ConeAlign] dot SVG(${svgX},${svgY}) videoLeft SVG x=${vx} top=${vy0} bottom=${vy1}`);
            // 三角形光锥：光点 -> 视频框左上角, 视频框左下角
            const svg = document.querySelector('.cone-svg');
            if (!svg) return;
            const midY = Math.round((vy0 + vy1) / 2);
            svg.innerHTML = `
                <defs>
                    <radialGradient id="coneGrad2" cx="${svgX}" cy="${svgY}" r="${Math.round((vx - svgX) * 2.5)}" gradientUnits="userSpaceOnUse">
                        <stop offset="0%" stop-color="#00e5ff" stop-opacity="0.8"/>
                        <stop offset="20%" stop-color="#00e5ff" stop-opacity="0.25"/>
                        <stop offset="50%" stop-color="#00e5ff" stop-opacity="0.08"/>
                        <stop offset="100%" stop-color="#00e5ff" stop-opacity="0.01"/>
                    </radialGradient>
                    <filter id="coneGlow2">
                        <feGaussianBlur stdDeviation="6" result="blur"/>
                        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                    </filter>
                </defs>
                <polygon points="${svgX},${svgY} ${vx},${vy0} ${vx},${vy1}" fill="url(#coneGrad2)" filter="url(#coneGlow2)"/>
                <line x1="${svgX}" y1="${svgY}" x2="${vx}" y2="${vy0}" stroke="rgba(0,229,255,0.25)" stroke-width="1.5"/>
                <line x1="${svgX}" y1="${svgY}" x2="${vx}" y2="${vy1}" stroke="rgba(0,229,255,0.25)" stroke-width="1.5"/>
                <circle cx="${svgX}" cy="${svgY}" r="5" fill="#00e5ff" opacity="0.9"/>
                <circle cx="${svgX}" cy="${svgY}" r="10" fill="none" stroke="#00e5ff" stroke-width="1.5" opacity="0.4">
                    <animate attributeName="r" values="10;18;10" dur="2s" repeatCount="indefinite"/>
                    <animate attributeName="opacity" values="0.4;0.1;0.4" dur="2s" repeatCount="indefinite"/>
                </circle>
                <circle r="3" fill="#00e5ff" opacity="0.6">
                    <animateMotion dur="1.5s" repeatCount="indefinite" path="M${svgX},${svgY} L${vx},${vy0}"/>
                </circle>
                <circle r="3" fill="#00e5ff" opacity="0.5">
                    <animateMotion dur="2s" repeatCount="indefinite" path="M${svgX},${svgY} L${vx},${vy1}" begin="0.5s"/>
                </circle>
                <circle r="2" fill="#00e5ff" opacity="0.4">
                    <animateMotion dur="1.8s" repeatCount="indefinite" path="M${svgX},${svgY} L${vx},${midY}" begin="0.3s"/>
                </circle>
            `;
        };
        if (boyImg.complete) { setTimeout(alignCone, 100); }
        else { boyImg.addEventListener('load', () => setTimeout(alignCone, 100)); }
        // 窗口大小改变时重新对齐
        window.addEventListener('resize', () => setTimeout(alignCone, 200));
    }
});

// ============================================================
// Live2D Avatar: TTS + Mouth Amp + Test (appended, no original code touched)
// ============================================================
function updateMouthAmp(val) {
    document.getElementById('mouth-amp-val').textContent = parseInt(val) + '%';
    if (typeof AvatarManager !== 'undefined') AvatarManager.setMouthAmplifier(parseInt(val) / 100);
}

async function testAvatar() {
    const text = (document.getElementById('avatar-test-text') || {}).value || '';
    if (!text) return;
    if (typeof AvatarManager !== 'undefined') {
        await AvatarManager.ensureUnlocked();
        // 根据文本情感设置表情
        AvatarManager.setExpression(_detectSentiment(text));
    }
    try {
        const resp = await fetch('/api/test_avatar?text=' + encodeURIComponent(text));
        const data = await resp.json();
        if (data.tts_audio_url && typeof AvatarManager !== 'undefined') {
            AvatarManager.speak(data.tts_audio_url);
            // 说完后回到 neutral
            const waitDone = () => {
                if (AvatarManager.isSpeaking) setTimeout(waitDone, 300);
                else AvatarManager.setExpression('neutral');
            };
            setTimeout(waitDone, 500);
            return;
        }
    } catch (e) {}
    if (typeof AvatarManager !== 'undefined') {
        AvatarManager.speakText(text);
        const waitDone = () => {
            if (AvatarManager.isSpeaking) setTimeout(waitDone, 300);
            else AvatarManager.setExpression('neutral');
        };
        setTimeout(waitDone, 500);
    }
}

// ============================================================
// Voice Input (Web Speech API — browser-native ASR)
// ============================================================
let _voiceRecognition = null;
let _isRecording = false;

function toggleVoiceInput() {
    if (_isRecording) {
        stopVoiceInput();
    } else {
        startVoiceInput();
    }
}

function startVoiceInput() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert('Your browser does not support speech recognition. Try Chrome.');
        return;
    }

    _voiceRecognition = new SpeechRecognition();
    _voiceRecognition.lang = 'en-US';
    _voiceRecognition.continuous = true;
    _voiceRecognition.interimResults = true;

    const btn = document.getElementById('btn-voice');
    const input = document.getElementById('chat-input');

    _voiceRecognition.onstart = () => {
        _isRecording = true;
        btn.classList.add('recording');
        btn.title = 'Listening... click to stop';
        input.placeholder = 'Listening... click mic again to stop';
        const boyMic = document.getElementById('btn-voice-boy');
        if (boyMic) boyMic.classList.add('recording');
    };

    _voiceRecognition.onresult = (event) => {
        let transcript = '';
        for (let i = 0; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
        }
        // 填入输入框，不自动发送，等用户确认后手动点 Send
        input.value = transcript;
    };

    _voiceRecognition.onerror = (event) => {
        console.warn('Speech recognition error:', event.error);
        if (event.error === 'not-allowed') {
            alert('Microphone access denied. Please allow microphone permission.');
        }
        stopVoiceInput();
    };

    _voiceRecognition.onend = () => {
        // continuous 模式下浏览器可能自动停止，仍在录音状态则重启
        if (_isRecording) {
            try { _voiceRecognition.start(); } catch(e) { stopVoiceInput(); }
        }
    };

    _voiceRecognition.start();
}

function stopVoiceInput() {
    _isRecording = false;
    const btn = document.getElementById('btn-voice');
    const input = document.getElementById('chat-input');
    if (btn) { btn.classList.remove('recording'); btn.title = 'Click to speak'; }
    const boyMic = document.getElementById('btn-voice-boy');
    if (boyMic) boyMic.classList.remove('recording');
    if (input && input.placeholder === 'Listening...') { input.placeholder = 'Type or use mic...'; }
    if (_voiceRecognition) {
        try { _voiceRecognition.stop(); } catch(e) {}
        _voiceRecognition = null;
    }
}

// ============================================================
// ASR Test (独立语音识别测试，只显示结果不发送)
// ============================================================
let _asrTestRecognition = null;
let _isASRTesting = false;

function toggleASRTest() {
    if (_isASRTesting) {
        stopASRTest();
    } else {
        startASRTest();
    }
}

function startASRTest() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert('Your browser does not support speech recognition. Try Chrome.');
        return;
    }

    const btn = document.getElementById('btn-asr-test');
    const status = document.getElementById('asr-test-status');
    const output = document.getElementById('asr-test-output');

    _asrTestRecognition = new SpeechRecognition();
    _asrTestRecognition.lang = 'en-US';
    _asrTestRecognition.continuous = true;
    _asrTestRecognition.interimResults = true;

    _asrTestRecognition.onstart = () => {
        _isASRTesting = true;
        if (btn) { btn.style.background = '#e74c3c'; btn.textContent = 'Stop ASR'; }
        if (status) status.textContent = 'Listening...';
        if (output) output.textContent = '';
    };

    _asrTestRecognition.onresult = (event) => {
        let interim = '';
        let final = '';
        for (let i = 0; i < event.results.length; i++) {
            if (event.results[i].isFinal) {
                final += event.results[i][0].transcript;
            } else {
                interim += event.results[i][0].transcript;
            }
        }
        if (output) {
            let html = '';
            if (final) html += '<span style="color:#2ecc71;">' + escapeHtml(final) + '</span> ';
            if (interim) html += '<span style="color:#f39c12; opacity:0.7;">' + escapeHtml(interim) + '</span>';
            output.innerHTML = html || '<span style="color:#9ca3b0;">...</span>';
        }
    };

    _asrTestRecognition.onerror = (event) => {
        console.warn('ASR Test error:', event.error);
        if (status) status.textContent = 'Error: ' + event.error;
        if (event.error === 'not-allowed') {
            alert('Microphone access denied. Please allow microphone permission.');
        }
        stopASRTest();
    };

    _asrTestRecognition.onend = () => {
        // continuous 模式下浏览器可能自动停止，如果还在测试状态则重启
        if (_isASRTesting) {
            try { _asrTestRecognition.start(); } catch(e) { stopASRTest(); }
        }
    };

    _asrTestRecognition.start();
}

function stopASRTest() {
    _isASRTesting = false;
    const btn = document.getElementById('btn-asr-test');
    const status = document.getElementById('asr-test-status');
    if (btn) { btn.style.background = '#e67e22'; btn.textContent = 'ASR Test'; }
    if (status) status.textContent = '(stopped)';
    if (_asrTestRecognition) {
        try { _asrTestRecognition.stop(); } catch(e) {}
        _asrTestRecognition = null;
    }
}
