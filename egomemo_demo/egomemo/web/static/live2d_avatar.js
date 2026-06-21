/**
 * Live2D Avatar Module for EgoMemo — Expression Engine v5
 *
 * 使用测试页面验证过的方案：替换 internalModel.update 跳过 motion manager
 */

const AvatarManager = (() => {
    let app = null;
    let model = null;
    let audioContext = null;
    let analyser = null;
    let isInitialized = false;
    let isSpeaking = false;
    let mouthTarget = 0;
    let mouthFormTarget = 0;
    let mouthAmplifier = 1.5;
    let lipSyncTimer = null;

    let cm = null;
    let paramIds = [];
    const forceParams = {};

    let lastTime = 0;
    let breathPhase = Math.random() * Math.PI * 2;
    let idlePhase = Math.random() * Math.PI * 2;
    let blinkTimer = 0;
    let nextBlinkTime = 2 + Math.random() * 4;
    let blinkProgress = 0;
    let isBlinking = false;
    let speakSwayPhase = 0;
    let speakEnergy = 0;
    let speakEnergySmoothed = 0;

    let currentExpression = 'neutral';
    const exprCurrent = {};

    // 表情+动作预设（身体动作幅度加大，让情绪从肢体也能看出来）
    const PRESETS = {
        neutral:   { ParamMouthForm: 1, ParamMouthOpenY: 0, ParamEyeLSmile: 0, ParamEyeRSmile: 0, ParamEyeLOpen: 1, ParamEyeROpen: 1, ParamBrowLY: 0, ParamBrowRY: 0, ParamBrowLAngle: 0, ParamBrowRAngle: 0, ParamCheek: 0, ParamAngleX: 0, ParamAngleY: 0, ParamAngleZ: 0, ParamBodyAngleX: 0, ParamBodyAngleY: 0, ParamBodyAngleZ: 0, ParamEyeBallX: 0, ParamEyeBallY: 0, ParamArmLA: -7, ParamArmRA: -5, ParamArmLB: 0, ParamArmRB: 0, ParamHandL: -0.6, ParamHandR: -0.6, ParamShoulder: 0, ParamLeg: 1 },
        happy:     { ParamMouthForm: 1, ParamMouthOpenY: 0.6, ParamEyeLSmile: 1, ParamEyeRSmile: 1, ParamEyeLOpen: 0.4, ParamEyeROpen: 0.4, ParamBrowLY: 0.8, ParamBrowRY: 0.8, ParamCheek: 1, ParamAngleX: 12, ParamAngleZ: -15, ParamBodyAngleX: 12, ParamBodyAngleZ: -8, ParamArmLA: 15, ParamArmRA: 15, ParamArmLB: 10, ParamArmRB: 10, ParamHandL: 3, ParamHandR: 3, ParamShoulder: 8, ParamLeg: 0.3 },
        excited:   { ParamMouthForm: 1, ParamMouthOpenY: 0.8, ParamEyeLSmile: 1, ParamEyeRSmile: 1, ParamEyeLOpen: 0.3, ParamEyeROpen: 0.3, ParamBrowLY: 0.8, ParamBrowRY: 0.8, ParamCheek: 1, ParamAngleX: 8, ParamAngleZ: -18, ParamBodyAngleX: 15, ParamBodyAngleZ: -6, ParamArmLA: 18, ParamArmRA: 18, ParamArmLB: 15, ParamArmRB: 15, ParamHandL: 4, ParamHandR: 4, ParamShoulder: 10, ParamLeg: 0.2 },
        angry:     { ParamMouthForm: -1, ParamMouthOpenY: 0.5, ParamEyeLSmile: 0, ParamEyeRSmile: 0, ParamEyeLOpen: 0.6, ParamEyeROpen: 0.6, ParamBrowLY: -1, ParamBrowRY: -1, ParamBrowLAngle: -1, ParamBrowRAngle: -1, ParamCheek: 0, ParamAngleX: -15, ParamAngleY: -15, ParamBodyAngleX: -12, ParamBodyAngleY: -5, ParamEyeBallY: -0.5, ParamArmLA: 12, ParamArmRA: 12, ParamArmLB: -10, ParamArmRB: -10, ParamHandL: -2, ParamHandR: -2, ParamShoulder: 8, ParamLeg: 0.8 },
        sad:       { ParamMouthForm: -1, ParamMouthOpenY: 0.1, ParamEyeLSmile: 0, ParamEyeRSmile: 0, ParamEyeLOpen: 0.3, ParamEyeROpen: 0.35, ParamBrowLY: -1, ParamBrowRY: -0.7, ParamBrowLAngle: 0.5, ParamBrowRAngle: 0.3, ParamCheek: 0, ParamAngleX: 15, ParamAngleY: -20, ParamAngleZ: 12, ParamBodyAngleX: -12, ParamBodyAngleY: -8, ParamBodyAngleZ: 6, ParamEyeBallY: -0.5, ParamArmLA: -10, ParamArmRA: -10, ParamArmLB: -3, ParamArmRB: -3, ParamHandL: -1, ParamHandR: -1, ParamShoulder: -4, ParamLeg: 0.8 },
        surprised: { ParamMouthForm: 0.3, ParamMouthOpenY: 1, ParamEyeLSmile: 0, ParamEyeRSmile: 0, ParamEyeLOpen: 1, ParamEyeROpen: 1, ParamBrowLY: 1, ParamBrowRY: 1, ParamCheek: 0.3, ParamAngleY: 15, ParamAngleX: -5, ParamBodyAngleY: 10, ParamBodyAngleX: -5, ParamEyeBallY: 0.3, ParamArmLA: 16, ParamArmRA: 16, ParamArmLB: 14, ParamArmRB: 14, ParamHandL: 4, ParamHandR: 4, ParamShoulder: 10, ParamLeg: 0.5 },
        warning:   { ParamMouthForm: -1, ParamMouthOpenY: 0.4, ParamEyeLSmile: 0, ParamEyeRSmile: 0, ParamEyeLOpen: 0.7, ParamEyeROpen: 0.7, ParamBrowLY: -1, ParamBrowRY: -1, ParamBrowLAngle: -1, ParamBrowRAngle: -1, ParamAngleY: -12, ParamAngleX: -8, ParamBodyAngleX: -8, ParamBodyAngleY: -5, ParamArmLA: 10, ParamArmRA: 10, ParamArmLB: -8, ParamArmRB: -8, ParamHandL: -1, ParamHandR: -1, ParamShoulder: 6, ParamLeg: 0.8 },
        concerned: { ParamMouthForm: -0.5, ParamMouthOpenY: 0, ParamEyeLSmile: 0, ParamEyeRSmile: 0, ParamEyeLOpen: 0.8, ParamEyeROpen: 0.8, ParamBrowLY: -0.6, ParamBrowRY: -0.4, ParamAngleX: 10, ParamAngleZ: 8, ParamBodyAngleX: -6, ParamBodyAngleZ: 4, ParamArmLA: -8, ParamArmRA: -8, ParamArmLB: -2, ParamArmRB: -2, ParamShoulder: -2, ParamLeg: 0.9 },
        thinking:  { ParamMouthForm: -0.3, ParamMouthOpenY: 0, ParamEyeLSmile: 0, ParamEyeRSmile: 0, ParamEyeLOpen: 0.7, ParamEyeROpen: 0.85, ParamBrowLY: 0.5, ParamBrowRY: -0.3, ParamCheek: 0, ParamAngleX: 20, ParamAngleY: 8, ParamAngleZ: 12, ParamBodyAngleX: 10, ParamBodyAngleZ: 5, ParamEyeBallX: 0.5, ParamEyeBallY: 0.3, ParamArmLA: -8, ParamArmRA: 12, ParamArmRB: 14, ParamHandR: 4, ParamShoulder: 3, ParamLeg: 0.9 },
        shy:       { ParamMouthForm: 0.5, ParamMouthOpenY: 0.2, ParamEyeLSmile: 0.6, ParamEyeRSmile: 0.6, ParamEyeLOpen: 0.5, ParamEyeROpen: 0.5, ParamBrowLY: -0.3, ParamBrowRY: -0.3, ParamCheek: 1, ParamAngleX: -18, ParamAngleZ: 15, ParamBodyAngleX: -15, ParamBodyAngleZ: 10, ParamEyeBallX: -0.5, ParamArmLA: -8, ParamArmRA: -8, ParamArmLB: 6, ParamArmRB: 6, ParamShoulder: 5, ParamLeg: 0.8 },
    };

    const MODEL_PATH = '/static/live2d/model/Hiyori/Hiyori.model3.json';

    function setParam(pid, value) {
        const idx = paramIds.indexOf(pid);
        if (idx >= 0) cm.setParameterValueByIndex(idx, value);
    }

    function lerp(a, b, t) { return a + (b - a) * t; }
    function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

    async function init(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return false;

        try {
            app = new PIXI.Application({
                view: canvas, autoStart: true, backgroundAlpha: 0,
                width: canvas.width, height: canvas.height,
                antialias: true,
                resolution: window.devicePixelRatio || 1,
                autoDensity: true,
            });

            model = await PIXI.live2d.Live2DModel.from(MODEL_PATH);

            const scale = Math.min(canvas.width / model.width, canvas.height / model.height) * 0.85;
            model.scale.set(scale);
            model.anchor.set(0.5, 0.5);
            model.x = canvas.width * 0.26;
            model.y = canvas.height * 0.38;
            app.stage.addChild(model);

            cm = model.internalModel.coreModel;
            paramIds = Array.from(cm._parameterIds);

            // 初始化表情插值状态
            const neutralPreset = PRESETS.neutral;
            for (const key of Object.keys(neutralPreset)) {
                exprCurrent[key] = neutralPreset[key];
            }

            // 默认状态
            forceParams['ParamEyeLOpen'] = 1;
            forceParams['ParamEyeROpen'] = 1;
            forceParams['ParamArmLA'] = -7;
            forceParams['ParamArmRA'] = -5;
            forceParams['ParamHandL'] = -0.62;
            forceParams['ParamHandR'] = -0.62;
            forceParams['ParamLeg'] = 1;
            forceParams['ParamMouthForm'] = 1;
            forceParams['ParamBreath'] = 0.5;
            forceParams['ParamHandLB'] = 10;
            forceParams['ParamHandRB'] = 10;

            // ★ 替换 internalModel.update，跳过 motion manager
            const im = model.internalModel;
            const origCMUpdate = cm.update.bind(cm);
            lastTime = performance.now();

            // 找到 PartArmA 和 PartArmB 的 part index，用于手动控制 opacity
            const partIds = cm._partIds;
            const partArmAIdx = partIds ? partIds.indexOf('PartArmA') : -1;
            const partArmBIdx = partIds ? partIds.indexOf('PartArmB') : -1;
            console.log('Arm part indices: A=' + partArmAIdx + ' B=' + partArmBIdx);

            im.update = function(dt, now) {
                const t = performance.now();
                const frameDt = clamp((t - lastTime) / 1000, 0.001, 0.1);
                lastTime = t;

                // 动画计算
                _animate(frameDt);

                // 应用所有强制参数
                for (const [pid, val] of Object.entries(forceParams)) {
                    setParam(pid, val);
                }

                // 物理模拟（头发、裙摆等）
                if (im.physics) im.physics.evaluate(cm, frameDt);

                // ★ 跳过 im.pose —— 它会把手臂部件互斥切换导致消失
                // 手动强制 ArmA 显示、ArmB 隐藏
                if (partArmAIdx >= 0) cm.setPartOpacityByIndex(partArmAIdx, 1);
                if (partArmBIdx >= 0) cm.setPartOpacityByIndex(partArmBIdx, 0);

                // 物理之后再次写入参数，防止被覆盖
                for (const [pid, val] of Object.entries(forceParams)) {
                    setParam(pid, val);
                }

                // 顶点计算
                origCMUpdate();
            };

            // Audio
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.4;

            isInitialized = true;
            setStatus('idle');
            console.log('Live2D avatar v5 ready. Params:', paramIds.length);
            return true;
        } catch (e) {
            console.error('Live2D init failed:', e);
            setStatus('error');
            return false;
        }
    }

    function _animate(dt) {
        // 呼吸
        breathPhase += dt * Math.PI * 2;
        forceParams['ParamBreath'] = (Math.sin(breathPhase) + 1) / 2 * 0.8;

        // 眨眼
        blinkTimer += dt;
        if (!isBlinking && blinkTimer >= nextBlinkTime) {
            isBlinking = true;
            blinkProgress = 0;
            blinkTimer = 0;
            nextBlinkTime = 2 + Math.random() * 5;
        }
        if (isBlinking) {
            blinkProgress += dt * 8;
            if (blinkProgress >= 1) {
                isBlinking = false;
            } else {
                const v = blinkProgress < 0.4 ? 1 - blinkProgress / 0.4 : (blinkProgress - 0.4) / 0.6;
                forceParams['ParamEyeLOpen'] = v;
                forceParams['ParamEyeROpen'] = v;
            }
        }
        if (!isBlinking) {
            // 眼睛开度由表情插值控制
            forceParams['ParamEyeLOpen'] = exprCurrent['ParamEyeLOpen'] || 1;
            forceParams['ParamEyeROpen'] = exprCurrent['ParamEyeROpen'] || 1;
        }

        // 嘴巴
        forceParams['ParamMouthOpenY'] = mouthTarget > 0.01 ? mouthTarget : (exprCurrent['ParamMouthOpenY'] || 0);

        // 不参与表情插值的参数（只锁定可见性相关，不锁动作）
        const LOCKED_PARAMS = new Set([
            'ParamHandLB', 'ParamHandRB',  // 手部可见性控制，必须保持 10
        ]);

        // 表情平滑插值（包括手臂/肩膀/腿部动作）
        const preset = PRESETS[currentExpression] || PRESETS.neutral;
        const rate = 0.10;
        for (const key of Object.keys(preset)) {
            if (key === 'ParamMouthOpenY') continue;  // 嘴巴开合由 TTS 控制
            if (LOCKED_PARAMS.has(key)) continue;      // 可见性控制保持不变
            let target = preset[key];
            if (key === 'ParamMouthForm' && isSpeaking) {
                target = lerp(target, mouthFormTarget, 0.6);
            }
            const prev = exprCurrent[key] !== undefined ? exprCurrent[key] : target;
            exprCurrent[key] = lerp(prev, target, rate);
            forceParams[key] = exprCurrent[key];
        }

        // 身体动态叠加（加大幅度，让肢体动作更明显）
        if (isSpeaking) {
            speakSwayPhase += dt * 3.5;
            speakEnergySmoothed = lerp(speakEnergySmoothed, speakEnergy, 0.15);
            const eBase = exprCurrent;
            // 身体扭转：基础表情值 + 说话摆动 + 能量驱动
            forceParams['ParamBodyAngleX'] = (eBase['ParamBodyAngleX'] || 0) + Math.sin(speakSwayPhase * 0.8) * 8 + Math.sin(speakSwayPhase * 1.7) * 3;
            forceParams['ParamBodyAngleY'] = (eBase['ParamBodyAngleY'] || 0) + speakEnergySmoothed * 8 + Math.sin(speakSwayPhase * 0.6) * 3;
            forceParams['ParamBodyAngleZ'] = (eBase['ParamBodyAngleZ'] || 0) + Math.sin(speakSwayPhase * 0.7) * 5;
            // 头部
            forceParams['ParamAngleX'] = (eBase['ParamAngleX'] || 0) + Math.sin(speakSwayPhase * 1.1) * 8 + Math.sin(speakSwayPhase * 2.1) * 3;
            forceParams['ParamAngleZ'] = (eBase['ParamAngleZ'] || 0) + Math.sin(speakSwayPhase * 0.9) * 6;
            forceParams['ParamAngleY'] = (eBase['ParamAngleY'] || 0) + speakEnergySmoothed * 5;
            // 肩膀随能量上下
            forceParams['ParamShoulder'] = (eBase['ParamShoulder'] || 0) + speakEnergySmoothed * 6 + Math.sin(speakSwayPhase * 1.5) * 3;
            // 腿部微动（说话时重心转移）
            forceParams['ParamLeg'] = (eBase['ParamLeg'] || 1) + Math.sin(speakSwayPhase * 0.5) * 0.3;
        } else {
            idlePhase += dt * 0.6;
            speakEnergySmoothed = lerp(speakEnergySmoothed, 0, 0.05);
            const eBase = exprCurrent;
            // 待机身体摇摆（幅度加大）
            forceParams['ParamBodyAngleX'] = (eBase['ParamBodyAngleX'] || 0) + Math.sin(idlePhase * 0.7) * 6 + Math.sin(idlePhase * 1.3) * 2.5;
            forceParams['ParamBodyAngleZ'] = (eBase['ParamBodyAngleZ'] || 0) + Math.sin(idlePhase * 0.5) * 4;
            forceParams['ParamBodyAngleY'] = (eBase['ParamBodyAngleY'] || 0) + Math.sin(idlePhase * 0.35) * 3;
            // 头部
            forceParams['ParamAngleX'] = (eBase['ParamAngleX'] || 0) + Math.sin(idlePhase * 0.4) * 7 + Math.sin(idlePhase * 0.9) * 3;
            forceParams['ParamAngleZ'] = (eBase['ParamAngleZ'] || 0) + Math.sin(idlePhase * 0.6) * 4;
            forceParams['ParamAngleY'] = (eBase['ParamAngleY'] || 0) + Math.sin(idlePhase * 0.3) * 4;
            // 眼球
            forceParams['ParamEyeBallX'] = (eBase['ParamEyeBallX'] || 0) + Math.sin(idlePhase * 0.25) * 0.4;
            forceParams['ParamEyeBallY'] = (eBase['ParamEyeBallY'] || 0) + Math.sin(idlePhase * 0.2) * 0.3;
            // 肩膀呼吸感
            forceParams['ParamShoulder'] = (eBase['ParamShoulder'] || 0) + Math.sin(idlePhase * 0.4) * 3;
            // 腿部重心微移
            forceParams['ParamLeg'] = (eBase['ParamLeg'] || 1) + Math.sin(idlePhase * 0.3) * 0.15;
        }

        // 手臂随身体摆动（幅度加大）
        if (isSpeaking) {
            const armBase = exprCurrent;
            forceParams['ParamArmLA'] = (armBase['ParamArmLA'] || -7) + Math.sin(speakSwayPhase * 0.7) * 10 + Math.sin(speakSwayPhase * 1.5) * 4;
            forceParams['ParamArmRA'] = (armBase['ParamArmRA'] || -5) + Math.sin(speakSwayPhase * 0.9) * 10 + Math.sin(speakSwayPhase * 1.3) * 4;
            // 手部随手臂
            forceParams['ParamHandL'] = (armBase['ParamHandL'] || -0.6) + Math.sin(speakSwayPhase * 1.1) * 2;
            forceParams['ParamHandR'] = (armBase['ParamHandR'] || -0.6) + Math.sin(speakSwayPhase * 1.3) * 2;
        } else {
            const armBase = exprCurrent;
            forceParams['ParamArmLA'] = (armBase['ParamArmLA'] || -7) + Math.sin(idlePhase * 0.6) * 7 + Math.sin(idlePhase * 1.2) * 3;
            forceParams['ParamArmRA'] = (armBase['ParamArmRA'] || -5) + Math.sin(idlePhase * 0.8) * 7 + Math.sin(idlePhase * 1.0) * 3;
            // 手部微动
            forceParams['ParamHandL'] = (armBase['ParamHandL'] || -0.6) + Math.sin(idlePhase * 0.5) * 1;
            forceParams['ParamHandR'] = (armBase['ParamHandR'] || -0.6) + Math.sin(idlePhase * 0.7) * 1;
        }
    }

    // ====== Audio ======

    async function ensureUnlocked() {
        if (!audioContext) return;
        if (audioContext.state === 'suspended') await audioContext.resume();
    }

    // 当前正在播的音频源；新的 speak() 前会先停掉它，防止多段 TTS 并发混播
    let currentAudioSource = null;

    /**
     * 播一段音频。返回 Promise，仅在音频 `onended`（真正播完）后 resolve。
     * 这样调用方可以 `await speak(url)` 实现严格串行，彻底避免 race。
     */
    async function speak(audioUrl) {
        if (!isInitialized || !audioUrl) return;
        if (audioContext.state === 'suspended') await audioContext.resume();

        // 如有在播的音频，先停掉（防并发混播）
        if (currentAudioSource) {
            try { currentAudioSource.onended = null; currentAudioSource.stop(0); } catch (_) {}
            currentAudioSource = null;
        }

        setStatus('speaking');

        let response, arrayBuffer, audioBuffer;
        try {
            response = await fetch(audioUrl);
            arrayBuffer = await response.arrayBuffer();
            audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        } catch (e) {
            console.error('Avatar speak fetch/decode error:', e);
            _stopSpeaking();
            return;
        }

        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(analyser);
        analyser.connect(audioContext.destination);
        currentAudioSource = source;

        // Promise 在 onended 时 resolve；用 resolveOnce 防止被多次调用
        return new Promise((resolve) => {
            let done = false;
            const resolveOnce = () => {
                if (done) return;
                done = true;
                resolve();
            };

            // 音频真正开始的这一刻，才让 Live2D 进入说话状态
            isSpeaking = true;
            speakSwayPhase = 0;
            source.start(0);

            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            const tick = () => {
                if (!isSpeaking) return;
                analyser.getByteFrequencyData(dataArray);

                let lowSum = 0;
                for (let i = 2; i < 15 && i < dataArray.length; i++) lowSum += dataArray[i];
                mouthTarget = clamp((lowSum / 13 / 50) * mouthAmplifier, 0, 1);

                let midSum = 0;
                for (let i = 15; i < 40 && i < dataArray.length; i++) midSum += dataArray[i];
                mouthFormTarget = clamp((midSum / 25 / 128 - 0.3) * 2.5, -0.8, 1.0);

                let totalSum = 0;
                for (let i = 2; i < 50 && i < dataArray.length; i++) totalSum += dataArray[i];
                speakEnergy = clamp(totalSum / 48 / 80, 0, 1);

                lipSyncTimer = requestAnimationFrame(tick);
            };
            lipSyncTimer = requestAnimationFrame(tick);

            source.onended = () => {
                if (currentAudioSource === source) currentAudioSource = null;
                _stopSpeaking();
                resolveOnce();
            };
            // 兜底：按音频时长也设个超时 resolve，防 onended 某些浏览器不触发
            const approxMs = Math.max(500, Math.ceil((audioBuffer.duration + 0.5) * 1000));
            setTimeout(resolveOnce, approxMs);
        });
    }

    /**
     * Web Speech API fallback。返回 Promise，仅在 onend 后 resolve。
     */
    function speakText(text) {
        if (!isInitialized || !text || !window.speechSynthesis) return Promise.resolve();

        setStatus('speaking');
        // 先停掉可能在播的其它 utterance
        try { window.speechSynthesis.cancel(); } catch (_) {}

        return new Promise((resolve) => {
            let done = false;
            const resolveOnce = () => { if (!done) { done = true; resolve(); } };

            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'en-US';

            let phase = 0;
            utterance.onstart = () => {
                isSpeaking = true;
                speakSwayPhase = 0;
                lipSyncTimer = setInterval(() => {
                    phase += 0.4;
                    mouthTarget = clamp((Math.sin(phase) + 1) / 2, 0, 1);
                    mouthFormTarget = Math.sin(phase * 0.7) * 0.6;
                    speakEnergy = clamp((Math.sin(phase * 0.5) + 1) / 3, 0, 1);
                }, 40);
            };
            utterance.onend = () => {
                clearInterval(lipSyncTimer);
                _stopSpeaking();
                resolveOnce();
            };
            utterance.onerror = () => {
                clearInterval(lipSyncTimer);
                _stopSpeaking();
                resolveOnce();
            };

            window.speechSynthesis.speak(utterance);
            // 兜底：按字数粗估时长超时 resolve
            setTimeout(resolveOnce, 1000 + text.length * 90);
        });
    }

    function _stopSpeaking() {
        isSpeaking = false;
        if (lipSyncTimer) {
            cancelAnimationFrame(lipSyncTimer);
            clearInterval(lipSyncTimer);
        }
        mouthTarget = 0;
        mouthFormTarget = 0;
        speakEnergy = 0;
        setStatus('idle');
    }

    function setExpression(name) {
        if (PRESETS[name]) currentExpression = name;
    }

    function setStatus(status) {
        const el = document.getElementById('avatar-status');
        if (!el) return;
        const labels = { idle: 'Idle', speaking: 'Speaking...', thinking: 'Thinking...', error: 'Avatar Error' };
        el.textContent = labels[status] || status;
        // 全息闪烁效果
        const overlay = document.getElementById('video-avatar-overlay');
        if (overlay) {
            overlay.classList.toggle('avatar-speaking', status === 'speaking');
        }
    }

    function setMouthAmplifier(val) { mouthAmplifier = val; }

    return {
        init, speak, speakText, setStatus, setMouthAmplifier,
        ensureUnlocked, setExpression,
        get isSpeaking() { return isSpeaking; },
    };
})();

document.addEventListener('DOMContentLoaded', () => {
    AvatarManager.init('live2d-canvas');

    function unlockAudio() {
        AvatarManager.ensureUnlocked();
        document.removeEventListener('click', unlockAudio);
        document.removeEventListener('keydown', unlockAudio);
        document.removeEventListener('touchstart', unlockAudio);
    }
    document.addEventListener('click', unlockAudio);
    document.addEventListener('keydown', unlockAudio);
    document.addEventListener('touchstart', unlockAudio);
});
