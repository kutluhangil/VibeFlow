import React, { useEffect, useRef, useState } from "react";
import { GoogleGenAI } from "@google/genai";
import {
  Camera,
  Square,
  Play,
  Music,
  Loader2,
  AlertCircle,
  Key,
  Activity,
  Cpu,
  ScanFace,
  Info,
  X,
  History,
  TrendingUp,
  Copy,
  Check,
  Heart,
  Lock,
  Unlock,
  Download,
  Disc,
  QrCode,
  Link,
  Shuffle,
  Timer,
} from "lucide-react";
import QRCode from "qrcode";
import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
} from "@mediapipe/tasks-vision";
import { motion, AnimatePresence } from "motion/react";
import * as Tone from "tone";
import { LineChart, Line, ResponsiveContainer, YAxis, Tooltip } from "recharts";

let hoverSynth: Tone.Synth | null = null;

async function initAudio() {
  if (Tone.context.state !== "running") {
    await Tone.start().catch(() => {});
  }
  if (!hoverSynth) {
    hoverSynth = new Tone.Synth({
      oscillator: { type: "sine" },
      envelope: { attack: 0.01, decay: 0.1, sustain: 0, release: 0.01 },
    }).toDestination();
    hoverSynth.volume.value = -15;
  }
}

declare global {
  interface Window {
    aistudio?: {
      hasSelectedApiKey: () => Promise<boolean>;
      openSelectKey: () => Promise<void>;
    };
  }
}

interface SmoothedBox {
  x: number;
  y: number;
  width: number;
  height: number;
  class: string;
  score: number;
  opacity: number;
  labelX: number;
  labelY: number;
}

class PCMPlayer {
  audioContext: AudioContext;
  nextStartTime: number;

  constructor(sampleRate: number = 48000) {
    this.audioContext = new (
      window.AudioContext || (window as any).webkitAudioContext
    )({ sampleRate });
    this.nextStartTime = this.audioContext.currentTime;
  }

  playChunk(base64Data: string) {
    const binaryString = atob(base64Data);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    // 16-bit PCM stereo
    const int16Array = new Int16Array(bytes.buffer);
    const numSamples = int16Array.length / 2;
    const leftChannel = new Float32Array(numSamples);
    const rightChannel = new Float32Array(numSamples);

    for (let i = 0; i < numSamples; i++) {
      leftChannel[i] = int16Array[i * 2] / 32768.0;
      rightChannel[i] = int16Array[i * 2 + 1] / 32768.0;
    }

    const audioBuffer = this.audioContext.createBuffer(
      2,
      numSamples,
      this.audioContext.sampleRate,
    );
    audioBuffer.getChannelData(0).set(leftChannel);
    audioBuffer.getChannelData(1).set(rightChannel);

    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioContext.destination);

    const currentTime = this.audioContext.currentTime;
    if (this.nextStartTime < currentTime) {
      this.nextStartTime = currentTime + 0.05;
    }

    source.start(this.nextStartTime);
    this.nextStartTime += audioBuffer.duration;
  }

  stop() {
    if (this.audioContext.state !== "closed") {
      this.audioContext.close();
    }
  }
}

import { ProceduralMusicEngine } from "./engine/ProceduralMusicEngine";
import { ThreeVisualizer } from "./components/ThreeVisualizer";
import { ActivityChart } from "./components/ActivityChart";

const VIBE_MAP: Record<string, string> = {
  person: "ethereal ambient drone, calm",
  "cell phone": "cyberpunk synthwave, electronic",
  laptop: "cyberpunk synthwave, electronic",
  tv: "cyberpunk synthwave, electronic",
  cup: "coffee shop jazz, chill acoustic",
  bottle: "coffee shop jazz, chill acoustic",
  bowl: "coffee shop jazz, chill acoustic",
  cat: "playful acoustic guitar, happy melody",
  dog: "playful acoustic guitar, happy melody",
  bird: "playful acoustic guitar, happy melody",
  car: "driving rock beat, fast tempo",
  bus: "driving rock beat, fast tempo",
  truck: "driving rock beat, fast tempo",
  chair: "ambient drone, relaxing",
  couch: "ambient drone, relaxing",
  bed: "ambient drone, relaxing",
  "potted plant": "ethereal flute, ambient nature",
  book: "classical piano, focused",
};

function getVibeForObjects(objects: string[]) {
  if (objects.length === 0) return "minimalist ambient drone, quiet";

  const vibes = new Set<string>();
  for (const obj of objects) {
    if (VIBE_MAP[obj]) {
      vibes.add(VIBE_MAP[obj]);
    } else {
      vibes.add("chill lofi beat");
    }
  }

  return Array.from(vibes).slice(0, 2).join(", ");
}

const getVibeFromGemini = async (
  objects: string[],
  emotion: string,
): Promise<string> => {
  try {
    const ai = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY || process.env.API_KEY,
    });
    const prompt = `You are a soundscape generator. Based on the following scene, output a 3-5 word ambient soundscape description (e.g., 'tribal rhythmic drone', 'cyberpunk electronic drone' or 'melancholy acoustic ambient'). Do not include any other text. Never output 'pop', 'upbeat', or 'energetic'. Everything must be ambient, but based on the expression. Scene: A person is feeling ${emotion} and the following objects are visible: ${objects.length > 0 ? objects.join(", ") : "none"}.`;
    const response = await ai.models.generateContent({
      model: "gemini-flash-lite-latest",
      contents: prompt,
    });
    return response.text?.trim() || "ambient drone, relaxing";
  } catch (e: any) {
    console.warn(
      "Gemini API error (falling back to local vibe map):",
      e.message || e,
    );
    return getVibeForObjects(objects) + `, ${emotion} mood`;
  }
};

import { Snapshot } from "./types";

export default function App() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [status, setStatus] = useState("Loading Object Detection Model...");
  const [currentPrompt, setCurrentPrompt] = useState("Waiting for camera...");
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [infoMsg, setInfoMsg] = useState<string | null>(null);
  const [isInfoOpen, setIsInfoOpen] = useState(false);
  const [presetVibe, setPresetVibe] = useState<string | null>(null);
  const presetVibeRef = useRef<string | null>(null);
  const [isPromptLocked, setIsPromptLocked] = useState(false);
  const isPromptLockedRef = useRef(false);
  const [isRecording, setIsRecording] = useState(false);
  const [forceScale, setForceScale] = useState<string | null>(null);
  const forceScaleRef = useRef<string | null>(null);
  const [globalBpm, setGlobalBpm] = useState<number | null>(null);
  const globalBpmRef = useRef<number | null>(null);
  const tapTimesRef = useRef<number[]>([]);
  const [isTapping, setIsTapping] = useState(false);
  const [isAutoCaptureActive, setIsAutoCaptureActive] = useState(false);
  const [autoCaptureInterval, setAutoCaptureInterval] = useState(30);
  const [timeUntilNextCapture, setTimeUntilNextCapture] = useState(30);
  const captureSnapshotRef = useRef<() => void>(() => {});
  const [delayAmount, setDelayAmount] = useState(0.4);
  const [delayTime, setDelayTime] = useState(0.33);
  const [intensity, setIntensity] = useState(1.0);
  const intensityRef = useRef(1.0);
  const [beatSync, setBeatSync] = useState(false);
  const beatSyncRef = useRef(false);
  const [autoGain, setAutoGain] = useState(false);
  const [crossfadeSpeed, setCrossfadeSpeed] = useState(0.02);
  const crossfadeSpeedRef = useRef(0.02);
  const proceduralEngineRef = useRef<ProceduralMusicEngine | null>(null);
  const [isAmbientLightMode, setIsAmbientLightMode] = useState(false);
  const isAmbientLightModeRef = useRef(false);
  const [isEmotionOverlayEnabled, setIsEmotionOverlayEnabled] = useState(false);
  const isEmotionOverlayEnabledRef = useRef(false);
  const [isHapticFeedbackEnabled, setIsHapticFeedbackEnabled] = useState(false);
  const isHapticFeedbackEnabledRef = useRef(false);
  const [vibrationIntensity, setVibrationIntensity] = useState(100);
  const vibrationIntensityRef = useRef(100);

  useEffect(() => {
    vibrationIntensityRef.current = vibrationIntensity;
  }, [vibrationIntensity]);
  const [ambientColor, setAmbientColor] = useState<[number, number, number]>([0, 0, 0]);
  const ambientColorRef = useRef<[number, number, number]>([0, 0, 0]);
  const avgCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    isAmbientLightModeRef.current = isAmbientLightMode;
    if (!isAmbientLightMode && containerRef.current) {
      containerRef.current.style.background = "";
    }
  }, [isAmbientLightMode]);

  useEffect(() => {
    isEmotionOverlayEnabledRef.current = isEmotionOverlayEnabled;
  }, [isEmotionOverlayEnabled]);

  useEffect(() => {
    isHapticFeedbackEnabledRef.current = isHapticFeedbackEnabled;
  }, [isHapticFeedbackEnabled]);

  const [visualizerMode, setVisualizerMode] = useState<
    "default" | "wireframe" | "particle"
  >("default");
  const visualizerModeRef = useRef<"default" | "wireframe" | "particle">(
    "default",
  );
  const [visualizerSensitivity, setVisualizerSensitivity] = useState(1.0);
  const visualizerSensitivityRef = useRef(1.0);
  const [hudTheme, setHudTheme] = useState<"white" | "amber" | "cyan">("white");
  const hudThemeRef = useRef<"white" | "amber" | "cyan">("white");
  const [recentVibes, setRecentVibes] = useState<string[]>([]);
  const [favoriteVibes, setFavoriteVibes] = useState<string[]>(() => {
    try {
      const saved = localStorage.getItem("favoriteVibes");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [chartData, setChartData] = useState<
    { time: string; activity: number }[]
  >([]);
  const [snapshots, setSnapshots] = useState<Snapshot[]>(() => {
    try {
      const saved = localStorage.getItem("snapshots");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [selectedQrSnapshot, setSelectedQrSnapshot] = useState<Snapshot | null>(null);
  const [qrDataUrl, setQrDataUrl] = useState<string>("");
  const [isQrTextCopied, setIsQrTextCopied] = useState(false);

  useEffect(() => {
    localStorage.setItem("favoriteVibes", JSON.stringify(favoriteVibes));
  }, [favoriteVibes]);

  useEffect(() => {
    localStorage.setItem("snapshots", JSON.stringify(snapshots));
  }, [snapshots]);
  const lastChartUpdateTimeRef = useRef<number>(0);
  const [consoleState, setConsoleState] = useState({
    emotion: "neutral",
    objects: [] as string[],
    blendshapes: {
      smile: 0,
      frown: 0,
      mouthOpen: 0,
      browRaise: 0,
      eyeBlink: 0,
      pucker: 0,
    },
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const faceCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sessionRef = useRef<any>(null);
  const playerRef = useRef<PCMPlayer | null>(null);

  const objectModelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const isPlayingRef = useRef(false);
  const lastPromptRef = useRef<string>("");
  const lastStateRef = useRef<string>("");
  const pendingStateRef = useRef<string | null>(null);
  const vibeTimeoutRef = useRef<any>(null);
  const lastStateUpdateTimeRef = useRef<number>(0);
  const lastBeatIndexRef = useRef<number>(0);
  const detectLoopRef = useRef<number | null>(null);
  const smoothedBoxesRef = useRef<Map<string, SmoothedBox>>(new Map());
  const smoothedBlendshapesRef = useRef({
    smile: 0,
    frown: 0,
    mouthOpen: 0,
    browRaise: 0,
    eyeBlink: 0,
    pucker: 0,
  });

  const handlePresetSelect = (
    presetLabel: string,
    presetValue: string | null,
  ) => {
    setPresetVibe(presetValue);
    presetVibeRef.current = presetValue;

    if (presetValue) {
      setCurrentPrompt(`[PRESET] ${presetLabel}: ${presetValue}`);
      if (sessionRef.current) {
        sessionRef.current
          .setWeightedPrompts({
            weightedPrompts: [{ text: presetValue, weight: 1.0 }],
          })
          .catch(console.error);
      }
      if (proceduralEngineRef.current) {
        proceduralEngineRef.current.setVibe(presetValue);
      }
    } else {
      // Re-trigger auto-detect
      lastStateRef.current = "";
      setCurrentPrompt("Auto-detecting vibe...");
    }
  };

  useEffect(() => {
    if (sessionRef.current && isPlaying) {
      sessionRef.current
        .setMusicGenerationConfig({
          musicGenerationConfig: { bpm: 120, temperature: intensity },
        })
        .catch(console.error);
    }
    if (proceduralEngineRef.current) {
      proceduralEngineRef.current.intensity = intensity;
    }
  }, [intensity, isPlaying]);

  useEffect(() => {
    if (proceduralEngineRef.current) {
      proceduralEngineRef.current.beatSync = beatSync;
    }
  }, [beatSync]);

  useEffect(() => {
    if (proceduralEngineRef.current) {
      proceduralEngineRef.current.crossfadeSpeed = crossfadeSpeed;
    }
  }, [crossfadeSpeed]);

  const [isCopied, setIsCopied] = useState(false);
  const [isLinkCopied, setIsLinkCopied] = useState(false);

  useEffect(() => {
    // Parse query parameters on mount to load shared session/vibe configurations
    const params = new URLSearchParams(window.location.search);
    const sharedVibe = params.get("vibe");
    if (sharedVibe) {
      setCurrentPrompt(sharedVibe);
      lastPromptRef.current = sharedVibe;
      setIsPromptLocked(true);
      isPromptLockedRef.current = true;

      const intensityParam = params.get("intensity");
      if (intensityParam) {
        const val = parseFloat(intensityParam);
        if (!isNaN(val)) {
          setIntensity(val);
          intensityRef.current = val;
        }
      }

      const csParam = params.get("crossfadeSpeed");
      if (csParam) {
        const val = parseFloat(csParam);
        if (!isNaN(val)) {
          setCrossfadeSpeed(val);
          crossfadeSpeedRef.current = val;
        }
      }

      const vsParam = params.get("visualizerSensitivity");
      if (vsParam) {
        const val = parseFloat(vsParam);
        if (!isNaN(val)) {
          setVisualizerSensitivity(val);
          visualizerSensitivityRef.current = val;
        }
      }

      const bpmParam = params.get("globalBpm");
      if (bpmParam) {
        if (bpmParam === "null") {
          setGlobalBpm(null);
          globalBpmRef.current = null;
        } else {
          const val = parseInt(bpmParam);
          if (!isNaN(val)) {
            setGlobalBpm(val);
            globalBpmRef.current = val;
          }
        }
      }

      const daParam = params.get("delayAmount");
      if (daParam) {
        const val = parseFloat(daParam);
        if (!isNaN(val)) {
          setDelayAmount(val);
        }
      }

      const dtParam = params.get("delayTime");
      if (dtParam) {
        const val = parseFloat(dtParam);
        if (!isNaN(val)) {
          setDelayTime(val);
        }
      }

      const bsParam = params.get("beatSync");
      if (bsParam) {
        const val = bsParam === "true";
        setBeatSync(val);
        beatSyncRef.current = val;
      }

      const agParam = params.get("autoGain");
      if (agParam) {
        const val = agParam === "true";
        setAutoGain(val);
      }

      const vmParam = params.get("visualizerMode");
      if (vmParam) {
        if (vmParam === "default" || vmParam === "wireframe" || vmParam === "particle") {
          setVisualizerMode(vmParam as any);
          visualizerModeRef.current = vmParam as any;
        }
      }

      const themeParam = params.get("hudTheme");
      if (themeParam) {
        if (themeParam === "white" || themeParam === "amber" || themeParam === "cyan") {
          setHudTheme(themeParam as any);
          hudThemeRef.current = themeParam as any;
        }
      }

      const fsParam = params.get("forceScale");
      if (fsParam) {
        setForceScale(fsParam);
        forceScaleRef.current = fsParam;
      }
    }
  }, []);

  const copyShareLinkToClipboard = () => {
    playHoverSound();
    const params = new URLSearchParams();
    if (currentPrompt) params.set("vibe", currentPrompt);
    params.set("intensity", intensity.toString());
    params.set("crossfadeSpeed", crossfadeSpeed.toString());
    params.set("visualizerSensitivity", visualizerSensitivity.toString());
    params.set("globalBpm", globalBpm === null ? "null" : globalBpm.toString());
    params.set("delayAmount", delayAmount.toString());
    params.set("delayTime", delayTime.toString());
    params.set("beatSync", beatSync ? "true" : "false");
    params.set("autoGain", autoGain ? "true" : "false");
    params.set("visualizerMode", visualizerMode);
    params.set("hudTheme", hudTheme);
    if (forceScale) params.set("forceScale", forceScale);

    const shareUrl = `${window.location.origin}${window.location.pathname}?${params.toString()}`;
    navigator.clipboard.writeText(shareUrl);
    setIsLinkCopied(true);
    setTimeout(() => setIsLinkCopied(false), 2000);
  };

  const copyVibeToClipboard = () => {
    playHoverSound();
    if (currentPrompt) {
      navigator.clipboard.writeText(currentPrompt);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    }
  };

  const [isShuffling, setIsShuffling] = useState(false);

  const handleShuffleVibe = async () => {
    playHoverSound();
    if (isShuffling) return;

    setIsShuffling(true);
    try {
      if (vibeTimeoutRef.current) {
        clearTimeout(vibeTimeoutRef.current);
      }

      const objects = consoleState.objects;
      const emotion = consoleState.emotion;

      const stateString = `${objects.join(",")}|${emotion}`;
      pendingStateRef.current = stateString;
      lastStateRef.current = stateString;

      const newVibe = await getVibeFromGemini(objects, emotion);

      setPresetVibe(null);
      presetVibeRef.current = null;
      setIsPromptLocked(false);
      isPromptLockedRef.current = false;

      lastPromptRef.current = newVibe;
      setCurrentPrompt(newVibe);
      setRecentVibes((prev) => {
        if (prev[0] === newVibe) return prev;
        return [newVibe, ...prev].slice(0, 10);
      });

      if (sessionRef.current) {
        await sessionRef.current
          .setWeightedPrompts({
            weightedPrompts: [{ text: newVibe, weight: 1.0 }],
          })
          .catch(console.error);
      }
      if (proceduralEngineRef.current) {
        proceduralEngineRef.current.setVibe(newVibe);
      }
    } catch (err) {
      console.error("Manual shuffle failed:", err);
    } finally {
      setIsShuffling(false);
    }
  };

  const handleTapTempo = () => {
    playHoverSound();
    setIsTapping(true);
    setTimeout(() => setIsTapping(false), 150);

    const now = Date.now();
    const lastTap = tapTimesRef.current[tapTimesRef.current.length - 1];

    if (lastTap && now - lastTap > 2500) {
      // Clear history if tap was too long ago (treating this tap as first of new sequence)
      tapTimesRef.current = [now];
    } else {
      tapTimesRef.current.push(now);
      if (tapTimesRef.current.length > 5) {
        tapTimesRef.current.shift();
      }
    }

    if (tapTimesRef.current.length > 1) {
      const diffs: number[] = [];
      for (let i = 1; i < tapTimesRef.current.length; i++) {
        diffs.push(tapTimesRef.current[i] - tapTimesRef.current[i - 1]);
      }
      const avgDiff = diffs.reduce((sum, d) => sum + d, 0) / diffs.length;
      const calculatedBpm = Math.round(60000 / avgDiff);
      
      // Limit to a reasonable range
      const finalBpm = Math.min(240, Math.max(40, calculatedBpm));
      setGlobalBpm(finalBpm);
      globalBpmRef.current = finalBpm;
      if (proceduralEngineRef.current) {
        proceduralEngineRef.current.setGlobalBpm(finalBpm);
      }
    }
  };

  const captureSnapshot = (isAuto: boolean = false) => {
    playHoverSound();
    if (!videoRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      const imageUri = canvas.toDataURL("image/jpeg", 0.8);
      const newSnapshot: Snapshot = {
        id: Date.now().toString(),
        timestamp: Date.now(),
        imageUri,
        audioProfile: currentPrompt || "No audio profile generated yet.",
        intensity,
        beatSync,
        autoGain,
        crossfadeSpeed,
        visualizerSensitivity,
        forceScale,
        globalBpm,
        isAutoCaptured: isAuto,
      };
      setSnapshots((prev) => [newSnapshot, ...prev]);
    }
  };

  useEffect(() => {
    captureSnapshotRef.current = () => captureSnapshot(true);
  }, [currentPrompt, intensity, beatSync, autoGain, crossfadeSpeed, visualizerSensitivity, forceScale, globalBpm]);

  // Auto-capture interval timer
  useEffect(() => {
    if (!isAutoCaptureActive || !isCameraActive) {
      setTimeUntilNextCapture(autoCaptureInterval);
      return;
    }

    setTimeUntilNextCapture(autoCaptureInterval);

    const timer = setInterval(() => {
      setTimeUntilNextCapture((prev) => {
        if (prev <= 1) {
          captureSnapshotRef.current();
          return autoCaptureInterval;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [isAutoCaptureActive, isCameraActive, autoCaptureInterval]);

  const handleGenerateQR = async (snap: Snapshot) => {
    setSelectedQrSnapshot(snap);
    setQrDataUrl("");
    try {
      const url = await QRCode.toDataURL(snap.audioProfile, {
        margin: 2,
        width: 300,
        color: {
          dark: "#000000",
          light: "#FFFFFF",
        },
      });
      setQrDataUrl(url);
    } catch (err) {
      console.error("Failed to generate QR code", err);
    }
  };

  const recallSnapshot = (snap: Snapshot) => {
    playHoverSound();

    setCurrentPrompt(snap.audioProfile);
    if (proceduralEngineRef.current) {
      proceduralEngineRef.current.setVibe(snap.audioProfile);
    }

    if (snap.intensity !== undefined) {
      setIntensity(snap.intensity);
      intensityRef.current = snap.intensity;
    }
    if (snap.beatSync !== undefined) {
      setBeatSync(snap.beatSync);
      beatSyncRef.current = snap.beatSync;
    }
    if (snap.autoGain !== undefined) {
      setAutoGain(snap.autoGain);
      if (proceduralEngineRef.current) {
        proceduralEngineRef.current.setAutoGain(snap.autoGain);
      }
    }
    if (snap.crossfadeSpeed !== undefined) {
      setCrossfadeSpeed(snap.crossfadeSpeed);
      crossfadeSpeedRef.current = snap.crossfadeSpeed;
    }
    if (snap.visualizerSensitivity !== undefined) {
      setVisualizerSensitivity(snap.visualizerSensitivity);
      visualizerSensitivityRef.current = snap.visualizerSensitivity;
    }
    if (snap.forceScale !== undefined) {
      setForceScale(snap.forceScale);
      forceScaleRef.current = snap.forceScale;
      if (proceduralEngineRef.current) {
        proceduralEngineRef.current.setForceScale(snap.forceScale);
      }
    }
    if (snap.globalBpm !== undefined) {
      setGlobalBpm(snap.globalBpm);
      globalBpmRef.current = snap.globalBpm;
      if (proceduralEngineRef.current) {
        proceduralEngineRef.current.setGlobalBpm(snap.globalBpm);
      }
    }

    setIsPromptLocked(true);
    isPromptLockedRef.current = true;
  };

  const playHoverSound = () => {
    try {
      initAudio();
      if (!hoverSynth || Tone.context.state !== "running") return;

      const now = Tone.now();
      hoverSynth.triggerAttackRelease(800, 0.1, now);
      hoverSynth.frequency.exponentialRampToValueAtTime(1200, now + 0.1);
    } catch (e) {}
  };

  useEffect(() => {
    const handleInteraction = () => initAudio();
    window.addEventListener("click", handleInteraction, { once: true });
    window.addEventListener("touchstart", handleInteraction, { once: true });
    return () => {
      window.removeEventListener("click", handleInteraction);
      window.removeEventListener("touchstart", handleInteraction);
    };
  }, []);

  useEffect(() => {
    // Load TensorFlow, COCO-SSD, and MediaPipe FaceLandmarker
    const loadModels = async () => {
      try {
        await tf.ready();
      } catch (err: any) {
        console.error("TF init failed:", err);
      }

      try {
        // Load COCO-SSD from the official Google TFJS models CDN
        const cocoModel = await cocoSsd.load();
        objectModelRef.current = cocoModel;
      } catch (err: any) {
        console.warn("COCO-SSD CDN load failed, attempting local fallback:", err);
        try {
          const cocoModel = await cocoSsd.load({ modelUrl: "/coco/model.json" });
          objectModelRef.current = cocoModel;
        } catch (localErr: any) {
          console.error("COCO-SSD local load failed:", localErr);
        }
      }

      try {
        // Use CDN for MediaPipe WebAssembly assets to ensure integrity
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm"
        );
        
        // Load FaceLandmarker model from official GCS bucket
        const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU",
          },
          outputFaceBlendshapes: true,
          runningMode: "VIDEO",
          numFaces: 1,
        });
        faceLandmarkerRef.current = faceLandmarker;

        // Load HandLandmarker model from official GCS bucket
        const handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
        });
        handLandmarkerRef.current = handLandmarker;

        setIsModelLoaded(true);
        setStatus("Idle");
      } catch (err: any) {
        console.error("Failed to load models:", err);
        setStatus("Error loading models");
        setErrorMsg(err.message);
      }
    };

    loadModels();

    return () => {
      stopSession();
    };
  }, []);

  const runDetection = async () => {
    if (
      !isPlayingRef.current ||
      !videoRef.current ||
      !canvasRef.current ||
      !objectModelRef.current
    )
      return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    if (video.readyState >= 2 && ctx) {
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      try {
        const predictions = await objectModelRef.current.detect(video);

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // --- Ambient Light Mode / Average Color Detection ---
        if (isAmbientLightModeRef.current) {
          if (!avgCanvasRef.current) {
            avgCanvasRef.current = document.createElement("canvas");
            avgCanvasRef.current.width = 1;
            avgCanvasRef.current.height = 1;
          }
          const avgCtx = avgCanvasRef.current.getContext("2d");
          if (avgCtx) {
            avgCtx.drawImage(video, 0, 0, 1, 1);
            try {
              const imgData = avgCtx.getImageData(0, 0, 1, 1).data;
              const targetR = imgData[0];
              const targetG = imgData[1];
              const targetB = imgData[2];

              const [prevR, prevG, prevB] = ambientColorRef.current;
              const lerpFactor = 0.05; // soft smoothing transition
              const nextR = prevR + (targetR - prevR) * lerpFactor;
              const nextG = prevG + (targetG - prevG) * lerpFactor;
              const nextB = prevB + (targetB - prevB) * lerpFactor;

              ambientColorRef.current = [nextR, nextG, nextB];
              
              if (containerRef.current) {
                // Apply a highly atmospheric gradient centered on the color, keeping the overall look dark/cyberpunk
                containerRef.current.style.background = `radial-gradient(circle at center, rgba(${Math.round(nextR)}, ${Math.round(nextG)}, ${Math.round(nextB)}, 0.18) 0%, rgba(0, 0, 0, 0.98) 100%)`;
              }
            } catch (e) {
              // Ignore cross-origin canvas errors if they occur
              console.warn("Could not calculate average color:", e);
            }
          }
        }

        const detectedClasses = new Set<string>();

        // --- Smoothing Logic ---
        const newSmoothedBoxes = new Map<string, SmoothedBox>();
        const unassignedPredictions = [...predictions];

        smoothedBoxesRef.current.forEach((box, id) => {
          let closestIdx = -1;
          let minDist = Infinity;
          unassignedPredictions.forEach((pred, idx) => {
            if (pred.class === box.class) {
              const [px, py, pw, ph] = pred.bbox;
              const dist = Math.hypot(
                px + pw / 2 - (box.x + box.width / 2),
                py + ph / 2 - (box.y + box.height / 2),
              );
              if (dist < 150) {
                if (dist < minDist) {
                  minDist = dist;
                  closestIdx = idx;
                }
              }
            }
          });

          if (closestIdx !== -1) {
            const pred = unassignedPredictions[closestIdx];
            const [px, py, pw, ph] = pred.bbox;
            const lerp = 0.15; // Smoothing factor
            box.x += (px - box.x) * lerp;
            box.y += (py - box.y) * lerp;
            box.width += (pw - box.width) * lerp;
            box.height += (ph - box.height) * lerp;
            box.opacity = Math.min(1, box.opacity + 0.1);
            box.score = pred.score;

            // Target label position (top right of box)
            const targetLabelX = box.x + box.width + 20;
            const targetLabelY = box.y - 20;
            box.labelX += (targetLabelX - box.labelX) * lerp;
            box.labelY += (targetLabelY - box.labelY) * lerp;

            newSmoothedBoxes.set(id, box);
            unassignedPredictions.splice(closestIdx, 1);
            detectedClasses.add(box.class);
          } else {
            box.opacity -= 0.05; // Fade out
            if (box.opacity > 0) {
              newSmoothedBoxes.set(id, box);
              detectedClasses.add(box.class);
            }
          }
        });

        unassignedPredictions.forEach((pred) => {
          const id = Math.random().toString(36).substring(7);
          const [x, y, width, height] = pred.bbox;
          newSmoothedBoxes.set(id, {
            x,
            y,
            width,
            height,
            class: pred.class,
            score: pred.score,
            opacity: 0,
            labelX: x + width + 40,
            labelY: y - 40,
          });
          detectedClasses.add(pred.class);
        });

        smoothedBoxesRef.current = newSmoothedBoxes;

        const themeRgbMap = {
          white: "255, 255, 255",
          amber: "255, 176, 0",
          cyan: "0, 240, 255",
        };
        const themeRgb = themeRgbMap[hudThemeRef.current];

        // --- Drawing Logic ---
        smoothedBoxesRef.current.forEach((box) => {
          const { x, y, width, height, opacity, labelX, labelY } = box;
          const text = `${box.class} (${Math.round(box.score * 100)}%)`;

          ctx.strokeStyle = `rgba(${themeRgb}, ${opacity * 0.8})`;
          ctx.lineWidth = 1;

          // Draw corners
          const cornerLength = Math.min(15, width / 4, height / 4);
          ctx.beginPath();
          ctx.moveTo(x, y + cornerLength);
          ctx.lineTo(x, y);
          ctx.lineTo(x + cornerLength, y);

          ctx.moveTo(x + width - cornerLength, y);
          ctx.lineTo(x + width, y);
          ctx.lineTo(x + width, y + cornerLength);

          ctx.moveTo(x + width, y + height - cornerLength);
          ctx.lineTo(x + width, y + height);
          ctx.lineTo(x + width - cornerLength, y + height);

          ctx.moveTo(x + cornerLength, y + height);
          ctx.lineTo(x, y + height);
          ctx.lineTo(x, y + height - cornerLength);
          ctx.stroke();

          // Crosshair center
          ctx.beginPath();
          ctx.moveTo(x + width / 2 - 5, y + height / 2);
          ctx.lineTo(x + width / 2 + 5, y + height / 2);
          ctx.moveTo(x + width / 2, y + height / 2 - 5);
          ctx.lineTo(x + width / 2, y + height / 2 + 5);
          ctx.strokeStyle = `rgba(${themeRgb}, ${opacity * 0.4})`;
          ctx.stroke();

          // Line to label
          ctx.beginPath();
          ctx.moveTo(x + width, y);
          ctx.lineTo(labelX, labelY + 16);
          ctx.strokeStyle = `rgba(${themeRgb}, ${opacity * 0.5})`;
          ctx.setLineDash([2, 2]);
          ctx.stroke();
          ctx.setLineDash([]);

          // Minimalist Label
          ctx.font = '400 10px "JetBrains Mono", monospace';
          const textWidth = ctx.measureText(text).width;
          ctx.fillStyle = `rgba(${themeRgb}, ${opacity * 0.2})`;
          ctx.fillRect(labelX, labelY, textWidth + 8, 16);
          ctx.fillStyle = `rgba(${themeRgb}, ${opacity})`;
          ctx.fillText(text.toUpperCase(), labelX + 4, labelY + 11);
        });

        // --- Spatial Audio Pan ---
        let avgX = 0;
        if (newSmoothedBoxes.size > 0) {
          newSmoothedBoxes.forEach((box) => {
            avgX += box.x + box.width / 2;
          });
          avgX /= newSmoothedBoxes.size;
          // Normalize to -1 to 1 (left to right)
          const pan = (avgX / canvas.width) * 2 - 1;
          if (proceduralEngineRef.current) {
            // Apply easing to pan to avoid sudden jumps
            const currentPan = proceduralEngineRef.current.spatialPan || 0;
            proceduralEngineRef.current.setSpatialPan(
              currentPan + (pan - currentPan) * 0.1,
            );
          }
        } else {
          if (proceduralEngineRef.current) {
            const currentPan = proceduralEngineRef.current.spatialPan || 0;
            proceduralEngineRef.current.setSpatialPan(currentPan * 0.9); // Return to center
          }
        }

        const classesArray = Array.from(detectedClasses).sort();

        let currentEmotion = "neutral";
        let currentBlendshapes = {
          smile: 0,
          frown: 0,
          mouthOpen: 0,
          browRaise: 0,
          eyeBlink: 0,
          pucker: 0,
        };

        if (faceLandmarkerRef.current) {
          const faceResult = faceLandmarkerRef.current.detectForVideo(
            video,
            performance.now(),
          );

          // Draw Face Mesh Point Cloud on secondary canvas
          if (faceCanvasRef.current) {
            const fCanvas = faceCanvasRef.current;
            const fCtx = fCanvas.getContext("2d");
            if (fCtx) {
              fCtx.clearRect(0, 0, fCanvas.width, fCanvas.height);

              if (
                faceResult.faceLandmarks &&
                faceResult.faceLandmarks.length > 0
              ) {
                const time = performance.now() / 1500; // 1.5 seconds per cycle

                // Find bounding box of face to center it
                let minX = video.videoWidth,
                  maxX = 0,
                  minY = video.videoHeight,
                  maxY = 0;
                for (const pt of faceResult.faceLandmarks[0]) {
                  const px = pt.x * video.videoWidth;
                  const py = pt.y * video.videoHeight;
                  if (px < minX) minX = px;
                  if (px > maxX) maxX = px;
                  if (py < minY) minY = py;
                  if (py > maxY) maxY = py;
                }
                const faceWidth = maxX - minX;
                const faceHeight = maxY - minY;
                const centerX = minX + faceWidth / 2;
                const centerY = minY + faceHeight / 2;

                const scanY = minY + ((Math.sin(time) + 1) / 2) * faceHeight;
                const scale =
                  Math.min(
                    fCanvas.width / faceWidth,
                    fCanvas.height / faceHeight,
                  ) * 0.8;

                // Subtle rhythmic pulse effect synchronizing with the current global BPM
                const bpm = globalBpmRef.current || 120;
                const beatDuration = (60 / bpm) * 1000;
                const beatProgress = (performance.now() % beatDuration) / beatDuration;
                // Elegant mathematical pulse curve peaking on the beat
                const pulse = Math.sin(beatProgress * Math.PI);

                // Draw standard point cloud with beat-pulsed intensity and theme-aligned highlights near the scan line
                for (const pt of faceResult.faceLandmarks[0]) {
                  const px = pt.x * video.videoWidth;
                  const py = pt.y * video.videoHeight;

                  // Distance from scan line (in pixels)
                  const dist = Math.abs(py - scanY) / faceHeight;
                  // Opacity: high near scan line, low elsewhere
                  const opacity = Math.max(0.15, 1.0 - dist * 4);

                  const drawX = fCanvas.width / 2 + (px - centerX) * scale;
                  const drawY = fCanvas.height / 2 + (py - centerY) * scale;

                  fCtx.beginPath();
                  if (dist < 0.15) {
                    // Highlight the scan line intersection with a glowing theme-colored overlay that pulses on the beat
                    fCtx.fillStyle = `rgba(${themeRgb}, ${Math.max(opacity, 0.4 + pulse * 0.4)})`;
                    fCtx.arc(drawX, drawY, 1.5 + pulse * 0.5, 0, 2 * Math.PI);
                  } else {
                    fCtx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
                    fCtx.arc(drawX, drawY, 1.5, 0, 2 * Math.PI);
                  }
                  fCtx.fill();
                }

                // Draw a beautiful glowing laser-like scan line overlay across the face width
                const drawScanY = fCanvas.height / 2 + (scanY - centerY) * scale;
                const faceMinDrawX = fCanvas.width / 2 + (minX - centerX) * scale;
                const faceMaxDrawX = fCanvas.width / 2 + (maxX - centerX) * scale;
                const paddingX = 15;
                const laserStartX = Math.max(5, faceMinDrawX - paddingX);
                const laserEndX = Math.min(fCanvas.width - 5, faceMaxDrawX + paddingX);

                fCtx.save();
                
                // Outer glowing laser layer (synced with the BPM pulse)
                fCtx.strokeStyle = `rgba(${themeRgb}, ${0.15 + pulse * 0.35})`;
                fCtx.lineWidth = 3.0 + pulse * 4.0;
                fCtx.shadowColor = `rgb(${themeRgb})`;
                fCtx.shadowBlur = 10.0 + pulse * 15.0;
                fCtx.beginPath();
                fCtx.moveTo(laserStartX, drawScanY);
                fCtx.lineTo(laserEndX, drawScanY);
                fCtx.stroke();

                // Inner bright core laser layer
                fCtx.strokeStyle = `rgba(255, 255, 255, ${0.8 + pulse * 0.2})`;
                fCtx.lineWidth = 1.0 + pulse * 0.5;
                fCtx.shadowBlur = 0;
                fCtx.beginPath();
                fCtx.moveTo(laserStartX, drawScanY);
                fCtx.lineTo(laserEndX, drawScanY);
                fCtx.stroke();

                // Draw clean status markers at scan extremities
                fCtx.fillStyle = `rgba(${themeRgb}, ${0.6 + pulse * 0.4})`;
                fCtx.font = 'bold 7px "JetBrains Mono", monospace';
                fCtx.fillText('BPM_LNK', laserStartX, drawScanY - 6);
                fCtx.fillText('SCAN_SYS', laserEndX - 36, drawScanY - 6);

                // Add circular beat-synchronized background sonar waves radiating from face center
                fCtx.strokeStyle = `rgba(${themeRgb}, ${(1.0 - beatProgress) * 0.15})`;
                fCtx.lineWidth = 1.0;
                fCtx.beginPath();
                fCtx.arc(fCanvas.width / 2, fCanvas.height / 2, beatProgress * 80, 0, 2 * Math.PI);
                fCtx.stroke();

                fCtx.restore();
              }
            }
          }

          if (
            faceResult.faceBlendshapes &&
            faceResult.faceBlendshapes.length > 0
          ) {
            const blendshapes = faceResult.faceBlendshapes[0].categories;
            const getScore = (name: string) =>
              blendshapes.find((b) => b.categoryName === name)?.score || 0;

            // Physical facial features for sliders
            currentBlendshapes.smile =
              (getScore("mouthSmileLeft") + getScore("mouthSmileRight")) / 2;
            currentBlendshapes.frown = Math.min(
              1,
              (getScore("mouthFrownLeft") +
                getScore("mouthFrownRight") +
                getScore("mouthRollLower")) *
                5,
            );
            currentBlendshapes.mouthOpen = getScore("jawOpen");
            currentBlendshapes.browRaise =
              (getScore("browInnerUp") +
                getScore("browOuterUpLeft") +
                getScore("browOuterUpRight")) /
              3;
            currentBlendshapes.eyeBlink =
              (getScore("eyeBlinkLeft") + getScore("eyeBlinkRight")) / 2;
            currentBlendshapes.pucker = getScore("mouthPucker");

            // High-level emotions for the overall state
            const surpriseScore =
              (getScore("jawOpen") + getScore("browInnerUp")) / 2;
            const angerScore =
              (getScore("browDownLeft") +
                getScore("browDownRight") +
                getScore("mouthPressLeft")) /
              3;
            const fearScore =
              ((getScore("jawOpen") +
                getScore("browInnerUp") +
                getScore("mouthStretchLeft") +
                getScore("mouthStretchRight")) /
                4) *
              0.6;
            const disgustScore = Math.min(
              1,
              (getScore("noseSneerLeft") +
                getScore("noseSneerRight") +
                getScore("mouthUpperUpLeft") +
                getScore("mouthUpperUpRight")) *
                4,
            );

            const emotions = [
              { name: "happy", score: currentBlendshapes.smile },
              { name: "sadness", score: currentBlendshapes.frown },
              { name: "surprised", score: surpriseScore },
              { name: "angry", score: angerScore },
              { name: "fear", score: fearScore },
              { name: "disgust", score: disgustScore },
            ];

            const maxEmotion = emotions.reduce(
              (max, e) => (e.score > max.score ? e : max),
              emotions[0],
            );
            if (maxEmotion.score > 0.2) {
              currentEmotion = maxEmotion.name;
            } else {
              currentEmotion = "neutral";
            }

            // --- Mobile Haptic Feedback Trigger ---
            const normalizedEmotion = currentEmotion.toLowerCase();
            const isHighIntensity = normalizedEmotion === "angry" || normalizedEmotion === "fear";
            if (isHapticFeedbackEnabledRef.current && isHighIntensity && typeof navigator !== "undefined" && navigator.vibrate) {
              const bpm = globalBpmRef.current || 120;
              const beatDuration = (60 / bpm) * 1000;
              const currentBeatIndex = Math.floor(performance.now() / beatDuration);

              if (currentBeatIndex !== lastBeatIndexRef.current) {
                lastBeatIndexRef.current = currentBeatIndex;
                const intensityVal = normalizedEmotion === "angry" ? angerScore : fearScore;
                if (intensityVal > 0.15) {
                  // Determine duration based on emotion intensity (range: 15ms - 100ms) scaled by vibrationIntensity
                  const intensityMultiplier = vibrationIntensityRef.current / 100;
                  const baseDuration = Math.min(200, Math.round((15 + intensityVal * 100) * intensityMultiplier));
                  if (baseDuration > 0) {
                    if (normalizedEmotion === "fear") {
                      // Double pulse for fear to simulate racing heartbeat
                      const gap = Math.round(50 * intensityMultiplier);
                      navigator.vibrate([baseDuration, gap, baseDuration]);
                    } else {
                      // Single crisp pulse for anger
                      navigator.vibrate(baseDuration);
                    }
                  }
                }
              }
            } else {
              // Keep beat tracking updated even if haptics is inactive or not high-intensity, so it doesn't instantly play a stale index when switched back
              const bpm = globalBpmRef.current || 120;
              const beatDuration = (60 / bpm) * 1000;
              lastBeatIndexRef.current = Math.floor(performance.now() / beatDuration);
            }
          }

          // Draw arrows and blurred line on main canvas
          if (faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
            const landmarks = faceResult.faceLandmarks[0];
            const mode = visualizerModeRef.current;

            // --- Minimalist Emotion Overlay ---
            if (isEmotionOverlayEnabledRef.current) {
              let minX = 1,
                maxX = 0,
                minY = 1,
                maxY = 0;
              for (const pt of landmarks) {
                if (pt.x < minX) minX = pt.x;
                if (pt.x > maxX) maxX = pt.x;
                if (pt.y < minY) minY = pt.y;
                if (pt.y > maxY) maxY = pt.y;
              }

              const faceCenterX = ((minX + maxX) / 2) * canvas.width;
              const faceTopY = minY * canvas.height;

              ctx.save();

              // Subtle micro-shake animation for high-intensity emotions like 'angry' or 'fear'
              const normalizedEmotion = currentEmotion.toLowerCase();
              const isHighIntensity = normalizedEmotion === "angry" || normalizedEmotion === "fear";
              const time = performance.now();

              if (isHighIntensity) {
                // Generate high-frequency, jittery displacement coordinates
                const shakeX = (Math.sin(time * 0.18) * 1.6) + (Math.cos(time * 0.43) * 0.8);
                const shakeY = (Math.cos(time * 0.22) * 1.3) + (Math.sin(time * 0.37) * 0.6);
                ctx.translate(shakeX, shakeY);
              }

              // Get the emotion string
              const emotionText = `STATE // ${currentEmotion.toUpperCase()}`;

              // Set font styles
              ctx.font = 'bold 11px "JetBrains Mono", monospace';
              const textWidth = ctx.measureText(emotionText).width;

              // Position directly above the user's face (with offset)
              const overlayX = faceCenterX - textWidth / 2;
              const overlayY = faceTopY - 30; // 30px above the face

              // Smooth beat-synchronized pulse for HUD elements
              const bpm = globalBpmRef.current || 120;
              const beatDuration = (60 / bpm) * 1000;
              const beatProgress = (time % beatDuration) / beatDuration;
              const pulse = Math.sin(beatProgress * Math.PI);
              const fastPulse = Math.sin(time * 0.03);

              // 1. Draw small pointing notch down towards head
              // Dynamic emotion color mapping
              let emotionColorRgb = themeRgb;
              switch (normalizedEmotion) {
                case "happy":
                  emotionColorRgb = "234, 179, 8"; // Gold/Yellow
                  break;
                case "sadness":
                  emotionColorRgb = "59, 130, 246"; // Blue
                  break;
                case "surprised":
                  emotionColorRgb = "168, 85, 247"; // Purple
                  break;
                case "angry":
                  emotionColorRgb = "239, 68, 68"; // Red
                  break;
                case "fear":
                  emotionColorRgb = "249, 115, 22"; // Orange
                  break;
                case "disgust":
                  emotionColorRgb = "132, 204, 22"; // Lime/Green
                  break;
                default:
                  emotionColorRgb = themeRgb;
                  break;
              }

              ctx.strokeStyle = `rgba(${emotionColorRgb}, ${0.5 + pulse * 0.2})`;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(faceCenterX, overlayY + 14);
              ctx.lineTo(faceCenterX, faceTopY - 10);
              ctx.stroke();

              // 2. Draw a minimalist capsule/background for the text
              // Background gets a subtle matching tint when an emotion is dominant
              ctx.fillStyle = normalizedEmotion !== "neutral"
                ? `rgba(${emotionColorRgb}, 0.12)`
                : "rgba(0, 0, 0, 0.75)";
              
              // Draw rounded rect or angular HUD box
              const boxX = overlayX - 10;
              const boxY = overlayY - 3;
              const boxW = textWidth + 20;
              const boxH = 20;
              
              ctx.beginPath();
              // Let's make an angular HUD style box (beveled corners)
              const bevel = 3;
              ctx.moveTo(boxX + bevel, boxY);
              ctx.lineTo(boxX + boxW - bevel, boxY);
              ctx.lineTo(boxX + boxW, boxY + bevel);
              ctx.lineTo(boxX + boxW, boxY + boxH - bevel);
              ctx.lineTo(boxX + boxW - bevel, boxY + boxH);
              ctx.lineTo(boxX + bevel, boxY + boxH);
              ctx.lineTo(boxX, boxY + boxH - bevel);
              ctx.lineTo(boxX, boxY + bevel);
              ctx.closePath();
              ctx.fill();

              // Draw primary border stroke
              ctx.strokeStyle = `rgba(${emotionColorRgb}, ${0.6 + pulse * 0.2})`;
              ctx.lineWidth = 1;
              ctx.stroke();

              // Subtle dynamic flash effect on the border (second glowing stroke with pulsing alpha)
              if (normalizedEmotion !== "neutral") {
                ctx.save();
                // Periodic high-frequency flash wave combined with rhythmic BPM pulse
                const flashVal = 0.5 + 0.5 * Math.sin(time * 0.008);
                ctx.strokeStyle = `rgba(${emotionColorRgb}, ${(0.2 + flashVal * 0.35) * (0.6 + pulse * 0.4)})`;
                ctx.lineWidth = 3.5;
                ctx.stroke();
                ctx.restore();
              }

              // 3. Draw a tiny pulsing status light inside the box
              ctx.fillStyle = `rgba(${emotionColorRgb}, ${0.8 + pulse * 0.2})`;
              ctx.beginPath();
              ctx.arc(boxX + 10, boxY + boxH / 2, 3, 0, Math.PI * 2);
              ctx.fill();

              // 4. Draw the text
              ctx.fillStyle = normalizedEmotion !== "neutral" ? "#ffffff" : "#f4f4f5";
              ctx.fillText(emotionText, boxX + 18, boxY + 13);

              ctx.restore();
            }

            if (mode === "default") {
              // Draw blurred line scanning over the face every 40 seconds
              const scanTime = performance.now() / 40000; // 40 seconds
              const scanPhase = scanTime % 1; // 0 to 1

              let minX = 1,
                maxX = 0,
                minY = 1,
                maxY = 0;
              for (const pt of landmarks) {
                if (pt.x < minX) minX = pt.x;
                if (pt.x > maxX) maxX = pt.x;
                if (pt.y < minY) minY = pt.y;
                if (pt.y > maxY) maxY = pt.y;
              }

              // Oscillate the scan line up and down
              const scanProgress = (Math.sin(scanPhase * Math.PI * 2) + 1) / 2; // 0 to 1 to 0
              const scanY = minY + scanProgress * (maxY - minY);

              // Opacity: 0 at top/bottom, 0.3 in the middle
              const lineOpacity = Math.sin(scanProgress * Math.PI) * 0.3;

              ctx.save();

              // Clip to face oval
              const faceOvalIndices = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
                365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
                132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
              ];
              ctx.beginPath();
              for (let i = 0; i < faceOvalIndices.length; i++) {
                const pt = landmarks[faceOvalIndices[i]];
                if (i === 0)
                  ctx.moveTo(pt.x * canvas.width, pt.y * canvas.height);
                else ctx.lineTo(pt.x * canvas.width, pt.y * canvas.height);
              }
              ctx.closePath();
              ctx.clip();

              if (lineOpacity > 0.01) {
                // 1. Draw the actual face landmarks illuminated by the scanner
                // This gives a highly realistic 3D contour effect without wacky math
                ctx.fillStyle = `rgb(${themeRgb})`; // White glow
                ctx.shadowColor = `rgb(${themeRgb})`;
                ctx.shadowBlur = 10;

                for (const pt of landmarks) {
                  // Calculate vertical distance from the scan line
                  const dist = Math.abs(pt.y - scanY);
                  const threshold = 0.04; // How thick the illuminated band is

                  if (dist < threshold) {
                    // Opacity falls off as points get further from the scan line
                    const ptOpacity = (1 - dist / threshold) * lineOpacity * 2;
                    ctx.globalAlpha = Math.min(ptOpacity, 1);

                    ctx.beginPath();
                    ctx.arc(
                      pt.x * canvas.width,
                      pt.y * canvas.height,
                      1.5,
                      0,
                      Math.PI * 2,
                    );
                    ctx.fill();
                  }
                }
                ctx.globalAlpha = 1.0;
              }

              ctx.restore();
            } else if (mode === "wireframe") {
              ctx.save();
              ctx.fillStyle = `rgba(${themeRgb}, 0.4)`;
              ctx.strokeStyle = `rgba(${themeRgb}, 0.1)`;
              ctx.lineWidth = 0.5;

              // Draw points and connect close ones
              for (let i = 0; i < landmarks.length; i += 2) {
                // Skip some points for performance and aesthetics
                const pt1 = landmarks[i];
                ctx.beginPath();
                ctx.arc(
                  pt1.x * canvas.width,
                  pt1.y * canvas.height,
                  1,
                  0,
                  Math.PI * 2,
                );
                ctx.fill();

                for (let j = i + 2; j < landmarks.length; j += 2) {
                  const pt2 = landmarks[j];
                  const dx = pt1.x - pt2.x;
                  const dy = pt1.y - pt2.y;
                  if (
                    dx * dx + dy * dy <
                    0.001 * visualizerSensitivityRef.current
                  ) {
                    ctx.beginPath();
                    ctx.moveTo(pt1.x * canvas.width, pt1.y * canvas.height);
                    ctx.lineTo(pt2.x * canvas.width, pt2.y * canvas.height);
                    ctx.stroke();
                  }
                }
              }
              ctx.restore();
            } else if (mode === "particle") {
              ctx.save();
              const time = performance.now() * 0.005;
              ctx.fillStyle = `rgba(${themeRgb}, 0.6)`;

              for (let i = 0; i < landmarks.length; i++) {
                const pt = landmarks[i];
                // Procedural noise-like offset
                const offsetX =
                  Math.sin(time + i) * 0.01 * visualizerSensitivityRef.current;
                const offsetY =
                  Math.cos(time * 0.8 + i) *
                  0.01 *
                  visualizerSensitivityRef.current;

                ctx.beginPath();
                ctx.arc(
                  (pt.x + offsetX) * canvas.width,
                  (pt.y + offsetY) * canvas.height,
                  1.5,
                  0,
                  Math.PI * 2,
                );
                ctx.fill();

                // Trail
                ctx.strokeStyle = `rgba(${themeRgb}, 0.2)`;
                ctx.beginPath();
                ctx.moveTo(pt.x * canvas.width, pt.y * canvas.height);
                ctx.lineTo(
                  (pt.x + offsetX) * canvas.width,
                  (pt.y + offsetY) * canvas.height,
                );
                ctx.stroke();
              }
              ctx.restore();
            }

            // Draw stylized feature highlights based on emotion
            const drawFeatureHighlight = (
              x: number,
              y: number,
              label: string,
              intensity: number,
            ) => {
              if (isNaN(intensity) || intensity < 0.05) return;
              ctx.save();
              ctx.translate(x * canvas.width, y * canvas.height);

              const size = 5 + intensity * 15;

              ctx.strokeStyle = `rgba(255, 255, 255, ${intensity * 0.8})`;
              ctx.lineWidth = 1.5;

              // Draw brackets [ ]
              ctx.beginPath();
              ctx.moveTo(-size, -size / 2);
              ctx.lineTo(-size, -size);
              ctx.lineTo(-size / 2, -size);

              ctx.moveTo(size, -size / 2);
              ctx.lineTo(size, -size);
              ctx.lineTo(size / 2, -size);

              ctx.moveTo(-size, size / 2);
              ctx.lineTo(-size, size);
              ctx.lineTo(-size / 2, size);

              ctx.moveTo(size, size / 2);
              ctx.lineTo(size, size);
              ctx.lineTo(size / 2, size);
              ctx.stroke();

              // Center dot
              ctx.fillStyle = `rgba(255, 255, 255, ${intensity})`;
              ctx.beginPath();
              ctx.arc(0, 0, 2, 0, Math.PI * 2);
              ctx.fill();

              // Label
              ctx.fillStyle = `rgba(255, 255, 255, ${intensity * 0.9})`;
              ctx.font = "10px monospace";
              ctx.fillText(label, size + 5, 3);

              ctx.restore();
            };

            const smoothed = smoothedBlendshapesRef.current;
            drawFeatureHighlight(
              landmarks[61].x,
              landmarks[61].y,
              "SMILE_L",
              smoothed.smile,
            );
            drawFeatureHighlight(
              landmarks[291].x,
              landmarks[291].y,
              "SMILE_R",
              smoothed.smile,
            );

            drawFeatureHighlight(
              landmarks[61].x,
              landmarks[61].y,
              "FROWN_L",
              smoothed.frown,
            );
            drawFeatureHighlight(
              landmarks[291].x,
              landmarks[291].y,
              "FROWN_R",
              smoothed.frown,
            );

            drawFeatureHighlight(
              landmarks[52].x,
              landmarks[52].y,
              "BROW_L",
              smoothed.browRaise,
            );
            drawFeatureHighlight(
              landmarks[282].x,
              landmarks[282].y,
              "BROW_R",
              smoothed.browRaise,
            );

            drawFeatureHighlight(
              landmarks[152].x,
              landmarks[152].y,
              "JAW_OPEN",
              smoothed.mouthOpen,
            );

            drawFeatureHighlight(
              landmarks[13].x,
              landmarks[13].y,
              "PUCKER",
              smoothed.pucker,
            );

            drawFeatureHighlight(
              landmarks[159].x,
              landmarks[159].y,
              "BLINK_L",
              smoothed.eyeBlink,
            );
            drawFeatureHighlight(
              landmarks[386].x,
              landmarks[386].y,
              "BLINK_R",
              smoothed.eyeBlink,
            );
          }
        }

        let handDetected = false;
        let handIndexTip: { x: number; y: number } | null = null;
        if (handLandmarkerRef.current) {
          const handResult = handLandmarkerRef.current.detectForVideo(
            video,
            performance.now(),
          );
          if (handResult.landmarks && handResult.landmarks.length > 0) {
            handDetected = true;
            handIndexTip = handResult.landmarks[0][8]; // Index finger tip

            const drawX = handIndexTip.x * canvas.width;
            const drawY = handIndexTip.y * canvas.height;

            ctx.save();
            ctx.strokeStyle = `rgb(${themeRgb})`;
            ctx.lineWidth = 1;
            ctx.shadowColor = `rgb(${themeRgb})`;
            ctx.shadowBlur = 10;
            ctx.beginPath();
            ctx.arc(drawX, drawY, 12, 0, 2 * Math.PI);
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(drawX, drawY, 4, 0, 2 * Math.PI);
            ctx.fill();

            // Draw axis lines for conductor mode
            ctx.setLineDash([2, 4]);
            ctx.beginPath();
            ctx.moveTo(drawX, 0);
            ctx.lineTo(drawX, canvas.height);
            ctx.moveTo(0, drawY);
            ctx.lineTo(canvas.width, drawY);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = `rgb(${themeRgb})`;
            ctx.font = 'bold 10px "JetBrains Mono"';
            ctx.fillText("CONDUCTOR ACTIVE", drawX + 20, drawY - 10);

            const mappedIntensity = Math.max(
              0.1,
              Math.min(2.0, 2.0 - handIndexTip.y * 1.9),
            );
            const mappedBpm = Math.max(
              40,
              Math.min(160, 40 + handIndexTip.x * 120),
            );
            ctx.fillText(
              `VOL:${mappedIntensity.toFixed(1)}x BPM:${Math.round(mappedBpm)}`,
              drawX + 20,
              drawY + 5,
            );
            ctx.restore();

            // Apply immediately to engine for responsiveness
            intensityRef.current = mappedIntensity;
            globalBpmRef.current = mappedBpm;
            if (proceduralEngineRef.current) {
              proceduralEngineRef.current.intensity = mappedIntensity;
              proceduralEngineRef.current.setGlobalBpm(mappedBpm);
            }
          }
        }

        // Throttle React state updates for the console UI to ~10fps
        const now = performance.now();
        if (now - lastStateUpdateTimeRef.current > 100) {
          const smoothingFactor = 0.15;
          const smoothed = smoothedBlendshapesRef.current;
          smoothed.smile +=
            (currentBlendshapes.smile - smoothed.smile) * smoothingFactor;
          smoothed.frown +=
            (currentBlendshapes.frown - smoothed.frown) * smoothingFactor;
          smoothed.mouthOpen +=
            (currentBlendshapes.mouthOpen - smoothed.mouthOpen) *
            smoothingFactor;
          smoothed.browRaise +=
            (currentBlendshapes.browRaise - smoothed.browRaise) *
            smoothingFactor;
          smoothed.eyeBlink +=
            (currentBlendshapes.eyeBlink - smoothed.eyeBlink) * smoothingFactor;
          smoothed.pucker +=
            (currentBlendshapes.pucker - smoothed.pucker) * smoothingFactor;

          setConsoleState({
            emotion: currentEmotion,
            objects: classesArray,
            blendshapes: { ...smoothed },
          });

          if (handDetected && handIndexTip) {
            const mappedIntensity = Math.max(
              0.1,
              Math.min(2.0, 2.0 - handIndexTip.y * 1.9),
            );
            const mappedBpm = Math.max(
              40,
              Math.min(160, 40 + handIndexTip.x * 120),
            );
            setIntensity(mappedIntensity);
            setGlobalBpm(mappedBpm);
          }

          lastStateUpdateTimeRef.current = now;
        }

        if (now - lastChartUpdateTimeRef.current > 2000) {
          const smoothed = smoothedBlendshapesRef.current;
          const activityScore =
            smoothed.smile * 2 +
            smoothed.frown * 2 +
            smoothed.mouthOpen +
            smoothed.browRaise * 1.5 +
            classesArray.length * 0.5;

          setChartData((prev) => {
            const timeStr = new Date().toLocaleTimeString([], {
              hour12: false,
              minute: "2-digit",
              second: "2-digit",
            });
            const newChart = [
              ...prev,
              { time: timeStr, activity: parseFloat(activityScore.toFixed(2)) },
            ];
            // Keep roughly 30 minutes of history if sampled every 2s? No, 30 min / 2s = 900 points.
            // Let's keep 60 points for visual clarity (2 minutes), or 900 if we want full 30 mins. Recharts handles lots of points ok, but maybe 150 points (5 mins). Let's go with 100 points for performance.
            if (newChart.length > 100) return newChart.slice(-100);
            return newChart;
          });
          lastChartUpdateTimeRef.current = now;
        }

        const stateString = `${classesArray.join(",")}|${currentEmotion}`;

        if (stateString !== pendingStateRef.current) {
          pendingStateRef.current = stateString;

          if (vibeTimeoutRef.current) {
            clearTimeout(vibeTimeoutRef.current);
          }

          // Debounce prompt changes by 3 seconds to avoid flickering
          vibeTimeoutRef.current = setTimeout(async () => {
            if (stateString !== lastStateRef.current) {
              lastStateRef.current = stateString;

              if (presetVibeRef.current) return;
              if (isPromptLockedRef.current) return;

              const newVibe = await getVibeFromGemini(
                classesArray,
                currentEmotion,
              );

              if (presetVibeRef.current) return;
              if (isPromptLockedRef.current) return;

              lastPromptRef.current = newVibe;
              setCurrentPrompt(newVibe);
              setRecentVibes((prev) => {
                // Avoid duplicate consecutive vibes in history
                if (prev[0] === newVibe) return prev;
                return [newVibe, ...prev].slice(0, 10);
              });

              if (sessionRef.current) {
                sessionRef.current
                  .setWeightedPrompts({
                    weightedPrompts: [{ text: newVibe, weight: 1.0 }],
                  })
                  .catch(console.error);
              }
              if (proceduralEngineRef.current) {
                proceduralEngineRef.current.setVibe(newVibe);
              }
            }
          }, 3000);
        }
      } catch (err) {
        console.error("Detection error:", err);
      }
    }

    if (isPlayingRef.current) {
      detectLoopRef.current = requestAnimationFrame(runDetection);
    }
  };

  const startSession = async () => {
    if (!isModelLoaded) return;

    try {
      setErrorMsg(null);
      setStatus("Starting camera...");

      let stream = streamRef.current;
      if (!stream) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: {
              width: { ideal: 1280 },
              height: { ideal: 720 },
              facingMode: "user",
            },
          });
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current
              .play()
              .catch((e) => console.error("Video play error:", e));
          }
          setIsCameraActive(true);
        } catch (camErr: any) {
          console.error("Camera error:", camErr);
          setStatus("Camera Error");
          setErrorMsg(
            "Camera access denied. Please allow camera access in your browser settings, then refresh the browser page.",
          );
          return;
        }
      }

      // Start detection immediately so it runs even if Lyria fails
      if (!isPlayingRef.current) {
        isPlayingRef.current = true;
        detectLoopRef.current = requestAnimationFrame(runDetection);

        if (!proceduralEngineRef.current) {
          proceduralEngineRef.current = new ProceduralMusicEngine();
        }
        proceduralEngineRef.current.intensity = intensity;
        proceduralEngineRef.current.beatSync = beatSync;
        proceduralEngineRef.current.crossfadeSpeed = crossfadeSpeed;
        proceduralEngineRef.current.start();
      }

      setStatus("Connecting to Lyria API...");
      playerRef.current = new PCMPlayer(48000);

      let timeoutId: any;

      let sessionPromise;
      try {
        const ai = new GoogleGenAI({
          apiKey:
            process.env.API_KEY || process.env.GEMINI_API_KEY || "dummy-key",
          apiVersion: "v1alpha",
        });

        sessionPromise = ai.live.music.connect({
          model: "lyria-realtime-exp",
          callbacks: {
            onmessage: (message: any) => {
              if (message.setupComplete) {
                console.log("Lyria setup complete");
              }
              const audioChunk = message.audioChunk;
              if (audioChunk?.data && playerRef.current) {
                playerRef.current.playChunk(audioChunk.data);
              }
            },
            onclose: () => {
              clearTimeout(timeoutId);
              if (isPlayingRef.current) {
                setInfoMsg("Lyria connection closed.");
                stopSession(false);
              } else {
                stopSession(false);
              }
            },
            onerror: (err: any) => {
              clearTimeout(timeoutId);
              console.error("Lyria API Error:", err);
              setErrorMsg(err.message || "Connection error with Lyria API.");
              setInfoMsg(null);
              stopSession(false);
            },
          },
        });
      } catch (err: any) {
        console.warn("Failed to initialize Lyria API:", err);
        clearTimeout(timeoutId);
        setErrorMsg("API key missing or invalid.");
        setInfoMsg(null);
        stopSession(false);
        return;
      }

      timeoutId = setTimeout(() => {
        setErrorMsg("Lyria API took too long to respond.");
        setInfoMsg(null);
        stopSession(false);
      }, 30000);

      sessionPromise
        .then(async (session) => {
          clearTimeout(timeoutId);
          sessionRef.current = session;
          setStatus("Connected & Playing");
          setIsPlaying(true);

          const initialPrompt = "minimalist ambient drone, quiet";
          setCurrentPrompt(initialPrompt);
          lastPromptRef.current = initialPrompt;

          try {
            await session.setMusicGenerationConfig({
              musicGenerationConfig: { bpm: 120, temperature: intensity },
            });
            await session.setWeightedPrompts({
              weightedPrompts: [{ text: initialPrompt, weight: 1.0 }],
            });
            session.play();
          } catch (e) {
            console.error("Error setting up session:", e);
          }
        })
        .catch((err) => {
          clearTimeout(timeoutId);
          console.error("API Error:", err);
          if (
            err.message?.includes("403") ||
            err.message?.includes("Permission denied") ||
            err.status === 403
          ) {
            setErrorMsg(
              "Lyria RealTime is experimental and requires allowlisting.",
            );
          } else {
            setErrorMsg(err.message || "An unknown API error occurred.");
          }
          setInfoMsg(null);
          stopSession(false);
        });
    } catch (err: any) {
      console.error("Setup Error:", err);
      setStatus("Failed to connect");
      setErrorMsg(err.message || "An unknown error occurred during setup.");
      setInfoMsg(null);
      stopSession(false);
    }
  };

  const stopSession = (closeCamera: boolean = true) => {
    setIsPlaying(false);
    if (vibeTimeoutRef.current) {
      clearTimeout(vibeTimeoutRef.current);
      vibeTimeoutRef.current = null;
    }
    pendingStateRef.current = null;

    if (
      status === "Connected & Playing" ||
      status === "Connecting to Lyria API..." ||
      status.includes("Local Synth")
    ) {
      setStatus("Idle");
    }

    if (playerRef.current) {
      playerRef.current.stop();
      playerRef.current = null;
    }
    if (sessionRef.current) {
      try {
        sessionRef.current.conn.close();
      } catch (e) {}
      sessionRef.current = null;
    }
    if (proceduralEngineRef.current) {
      proceduralEngineRef.current.stop();
      proceduralEngineRef.current = null;
    }

    setConsoleState({
      emotion: "neutral",
      objects: [],
      blendshapes: {
        smile: 0,
        frown: 0,
        mouthOpen: 0,
        browRaise: 0,
        eyeBlink: 0,
        pucker: 0,
      },
    });
    smoothedBlendshapesRef.current = {
      smile: 0,
      frown: 0,
      mouthOpen: 0,
      browRaise: 0,
      eyeBlink: 0,
      pucker: 0,
    };
    smoothedBoxesRef.current.clear();

    setInfoMsg(null);

    if (closeCamera) {
      isPlayingRef.current = false;
      setCurrentPrompt("Waiting for camera...");

      if (detectLoopRef.current) {
        cancelAnimationFrame(detectLoopRef.current);
        detectLoopRef.current = null;
      }

      // Let the canvas fade out via CSS transition instead of clearing immediately
      // if (canvasRef.current) {
      //   const ctx = canvasRef.current.getContext('2d');
      //   if (ctx) {
      //     ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      //   }
      // }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        setIsCameraActive(false);
      }
    }
  };

  const themeHexMap = {
    white: "#ffffff",
    amber: "#ffb000",
    cyan: "#00f0ff",
  };
  const themeColor = themeHexMap[hudTheme];

  const themeRgbMap = {
    white: "255, 255, 255",
    amber: "255, 176, 0",
    cyan: "0, 240, 255",
  };

  return (
    <div
      ref={containerRef}
      className={`h-[100dvh] w-full bg-black text-white flex overflow-hidden font-mono relative theme-${hudTheme}`}
    >
      {/* Background Camera Feed */}
      <div className="absolute inset-0 z-0">
        {!isCameraActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white/50 z-10 font-mono text-sm">
            <Camera className="w-8 h-8 mb-4 opacity-50" />
            <p>SYSTEM.CAMERA_OFFLINE</p>
          </div>
        )}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`absolute inset-0 w-full h-full object-cover grayscale contrast-125 opacity-60 transition-opacity duration-500 ${isCameraActive ? "opacity-100" : "opacity-0"}`}
        />
        <ThreeVisualizer
          engine={proceduralEngineRef.current}
          mode={visualizerMode}
          sensitivity={visualizerSensitivity}
          themeColor={themeColor}
        />
        <canvas
          ref={canvasRef}
          className={`absolute inset-0 w-full h-full object-cover pointer-events-none transition-opacity duration-500 z-[15] ${isCameraActive ? "opacity-100" : "opacity-0"}`}
        />
        {/* Vignette & Scanlines */}
        <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.8)_100%)] z-10" />
        <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(transparent_50%,rgba(0,0,0,0.25)_50%)] bg-[length:100%_4px] z-10" />

        {/* Decorative HUD Elements */}
        <div className="absolute inset-0 pointer-events-none z-10 flex items-center justify-center overflow-hidden">
          <div className="w-[150vw] h-[150vw] sm:w-[600px] sm:h-[600px] border border-white/10 rounded-full border-dashed animate-[spin_60s_linear_infinite] shrink-0" />
          <div className="absolute w-[100vw] h-[100vw] sm:w-[400px] sm:h-[400px] border border-white/5 rounded-full animate-[spin_40s_linear_infinite_reverse] shrink-0" />
          <div className="absolute w-px h-full bg-white/5" />
          <div className="absolute h-px w-full bg-white/5" />
        </div>
      </div>

      {/* Overlays */}
      <div className="relative z-20 w-full h-full pointer-events-auto p-4 sm:p-6 overflow-y-auto overflow-x-hidden pb-32 sm:pb-6">
        <div className="flex flex-col lg:flex-row justify-between gap-4 min-h-full">
          {/* Left Column */}
          <div className="contents lg:flex lg:flex-col lg:justify-between w-full lg:w-80 pointer-events-none shrink-0">
            {/* Top Left: System Status & Mobile Controls */}
            <div className="flex flex-col gap-4 shrink-0 order-1 lg:order-none pointer-events-auto">
              {/* Status */}
              <div className="flex flex-col items-start gap-4 shrink-0">
                <div className="flex items-start justify-between w-full">
                  <div>
                    <h1 className="text-2xl font-bold tracking-tighter text-white drop-shadow-[0_0_8px_rgba(255,255,255,0.8)]">
                      VISION_SYNC
                    </h1>
                    <p className="text-[10px] text-white/70 font-mono uppercase tracking-widest">
                      Lyria RealTime Engine v2.4
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    {/* Small Start/Stop Button (Mobile Landscape Only) */}
                    <button
                      onClick={() => {
                        playHoverSound();
                        isCameraActive ? stopSession(true) : startSession();
                      }}
                      onMouseEnter={playHoverSound}
                      disabled={!isModelLoaded}
                      className={`hidden landscape:flex lg:landscape:hidden justify-center items-center gap-2 px-3 py-2 font-mono text-[10px] font-bold uppercase tracking-widest transition-all duration-300 border backdrop-blur-md rounded-none ${
                        isCameraActive
                          ? "bg-red-500/20 text-red-400 border-red-500 hover:bg-red-500/30 shadow-[0_0_10px_rgba(239,68,68,0.4)]"
                          : "bg-white/10 text-white border-white hover:bg-white/20 shadow-[0_0_10px_rgba(255,255,255,0.3)]"
                      } disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                      {!isModelLoaded ? (
                        <>
                          <Loader2 className="w-3 h-3 animate-spin" /> INIT
                        </>
                      ) : isCameraActive ? (
                        <>
                          <Square className="w-3 h-3 fill-current" /> STOP
                        </>
                      ) : (
                        <>
                          <Play className="w-3 h-3 fill-current" /> START
                        </>
                      )}
                    </button>
                    <button
                      onClick={() => {
                        playHoverSound();
                        setIsInfoOpen(true);
                      }}
                      onMouseEnter={playHoverSound}
                      className="p-2 bg-white/10 hover:bg-white/20 rounded-full transition-colors backdrop-blur-md border border-white/20 shrink-0"
                      title="App Information"
                    >
                      <Info className="w-5 h-5 text-white" />
                    </button>
                  </div>
                </div>
                <div className="text-xs font-mono text-white/80 flex items-center gap-2 bg-black/40 backdrop-blur px-3 py-1.5 border border-white/20">
                  <div
                    className={`w-2 h-2 rounded-none ${status === "Connected & Playing" ? "bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)]" : status.includes("Connecting") || status.includes("Starting") ? "bg-yellow-500 shadow-[0_0_8px_rgba(234,179,8,0.8)]" : status === "Loading Object Detection Model..." ? "bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.8)]" : status.includes("Error") || status.includes("Denied") ? "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]" : "bg-zinc-600"}`}
                  />
                  {status}
                </div>
              </div>

              {/* Mobile Controls (Hidden on Desktop & Landscape) */}
              <div className="flex lg:hidden landscape:hidden flex-col items-stretch gap-4 shrink-0">
                <button
                  onClick={() => {
                    playHoverSound();
                    isCameraActive ? stopSession(true) : startSession();
                  }}
                  onMouseEnter={playHoverSound}
                  disabled={!isModelLoaded}
                  className={`flex justify-center items-center gap-3 px-10 py-4 font-mono text-sm font-bold uppercase tracking-widest transition-all duration-300 border-2 backdrop-blur-md ${
                    isCameraActive
                      ? "bg-red-500/20 text-red-400 border-red-500 hover:bg-red-500/30 shadow-[0_0_20px_rgba(239,68,68,0.4)]"
                      : "bg-white/10 text-white border-white hover:bg-white/20 shadow-[0_0_20px_rgba(255,255,255,0.3)]"
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  {!isModelLoaded ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />{" "}
                      INITIALIZING...
                    </>
                  ) : isCameraActive ? (
                    <>
                      <Square className="w-5 h-5 fill-current" /> STOP SYSTEM
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5 fill-current" /> START SYSTEM
                    </>
                  )}
                </button>
              </div>

              {/* Mobile Camera Viewport Spacer */}
              <div className="h-[45vh] landscape:h-[100vh] lg:hidden pointer-events-none shrink-0" />
            </div>

            {/* Bottom Left: Scan & Affective */}
            <div className="flex flex-col landscape:flex-row lg:landscape:flex-col gap-4 shrink-0 lg:mt-auto order-4 lg:order-none pointer-events-auto">
              {/* Middle Left: Face Scanner */}
              <div className="flex flex-col justify-center shrink-0 landscape:flex-1 lg:landscape:flex-none">
                <motion.div
                  key={`biometric-scan-${globalBpm || 120}`}
                  animate={{
                    scale: isCameraActive ? [1, 1.015, 1] : 1,
                  }}
                  transition={{
                    duration: 60 / (globalBpm || 120),
                    repeat: isCameraActive ? Infinity : 0,
                    ease: "easeInOut",
                  }}
                  className="biometric-scan-panel bg-black/40 backdrop-blur-md border border-white/20 p-4 w-full shadow-[0_0_30px_rgba(0,0,0,0.8)] relative overflow-hidden flex flex-col h-64 landscape:h-full lg:landscape:h-64 shrink-0"
                  title="Real-time facial landmark tracking"
                >
                  <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-2 shrink-0 flex items-center gap-2">
                    <ScanFace className="w-3 h-3" />
                    Biometric Scan
                  </h3>
                  <div className="relative w-full flex-1 border border-white/10 flex items-center justify-center bg-white/5 min-h-0">
                    <canvas
                      ref={faceCanvasRef}
                      width={300}
                      height={300}
                      className={`w-full h-full object-contain transition-opacity duration-500 ${isCameraActive ? "opacity-100" : "opacity-0"}`}
                    />
                  </div>
                </motion.div>
              </div>

              {/* Bottom Left: Affective State */}
              <div className="flex flex-col justify-end shrink-0 landscape:flex-1 lg:landscape:flex-none">
                <div
                  className="bg-black/40 backdrop-blur-md border border-white/20 p-5 w-full h-full shadow-[0_0_30px_rgba(0,0,0,0.8)]"
                  title="Detected emotional state based on facial expressions"
                >
                  <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-3 flex items-center gap-2">
                    <Activity className="w-3 h-3" />
                    Affective State
                  </h3>
                  <div className="text-3xl font-light tracking-tighter mb-4 capitalize text-white drop-shadow-[0_0_8px_rgba(255,255,255,0.5)]">
                    {consoleState.emotion}
                  </div>

                  <div className="space-y-2">
                    {[
                      { label: "Smile", value: consoleState.blendshapes.smile },
                      { label: "Frown", value: consoleState.blendshapes.frown },
                      {
                        label: "Mouth Open",
                        value: consoleState.blendshapes.mouthOpen,
                      },
                      {
                        label: "Brow Raise",
                        value: consoleState.blendshapes.browRaise,
                      },
                      {
                        label: "Eye Blink",
                        value: consoleState.blendshapes.eyeBlink,
                      },
                    ].map((item) => (
                      <div key={item.label}>
                        <div className="flex justify-between text-[10px] mb-1">
                          <span className="text-white/60 uppercase tracking-wider">
                            {item.label}
                          </span>
                          <span className="font-bold text-white/90">
                            {isNaN(item.value)
                              ? 0
                              : (item.value * 100).toFixed(0)}
                            %
                          </span>
                        </div>
                        <div className="h-[2px] bg-white/10 overflow-hidden">
                          <motion.div
                            className="h-full bg-white shadow-[0_0_10px_rgba(255,255,255,0.8)]"
                            initial={{ width: 0 }}
                            animate={{
                              width: `${isNaN(item.value) ? 0 : item.value * 100}%`,
                            }}
                            transition={{
                              type: "spring",
                              bounce: 0,
                              duration: 0.5,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="contents lg:flex lg:flex-col lg:justify-between lg:items-end w-full lg:w-80 pointer-events-none shrink-0 mt-0">
            {/* Top Right: Controls */}
            <div className="hidden lg:flex flex-col items-end gap-4 shrink-0 order-none pointer-events-auto">
              <button
                onClick={() => {
                  playHoverSound();
                  isCameraActive ? stopSession(true) : startSession();
                }}
                onMouseEnter={playHoverSound}
                disabled={!isModelLoaded}
                className={`flex justify-center items-center gap-3 px-10 py-4 font-mono text-sm font-bold uppercase tracking-widest transition-all duration-300 border-2 backdrop-blur-md ${
                  isCameraActive
                    ? "bg-red-500/20 text-red-400 border-red-500 hover:bg-red-500/30 shadow-[0_0_20px_rgba(239,68,68,0.4)]"
                    : "bg-white/10 text-white border-white hover:bg-white/20 shadow-[0_0_20px_rgba(255,255,255,0.3)]"
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {!isModelLoaded ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" /> INITIALIZING...
                  </>
                ) : isCameraActive ? (
                  <>
                    <Square className="w-5 h-5 fill-current" /> STOP SYSTEM
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5 fill-current" /> START SYSTEM
                  </>
                )}
              </button>
            </div>

            {/* Bottom Right: Entities & Audio */}
            <div className="flex flex-col gap-4 shrink-0 w-full order-2 lg:order-none pointer-events-auto">
              {/* Detected Entities */}
              <div
                className="order-1 lg:order-2 w-full bg-black/40 backdrop-blur-md border border-white/20 p-5 shadow-[0_0_30px_rgba(0,0,0,0.8)]"
                title="Objects detected in the camera view"
              >
                <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <Cpu className="w-3 h-3" />
                  Entities
                </h3>
                {consoleState.objects.length === 0 ? (
                  <p className="text-[10px] text-white/40 italic">
                    No entities detected.
                  </p>
                ) : (
                  <ul className="space-y-1.5">
                    <AnimatePresence>
                      {consoleState.objects.map((obj) => (
                        <motion.li
                          key={obj}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: 10 }}
                          className="text-[10px] flex items-center gap-2 text-white/90 uppercase tracking-wider"
                        >
                          <span className="w-1 h-1 bg-white shadow-[0_0_5px_rgba(255,255,255,0.8)]" />
                          {obj}
                        </motion.li>
                      ))}
                    </AnimatePresence>
                  </ul>
                )}
              </div>

              {/* Vibe Presets & Controls */}
              <div
                className="order-2 lg:order-1 w-full bg-black/40 backdrop-blur-md border border-white/20 p-5 shadow-[0_0_30px_rgba(0,0,0,0.8)]"
                title="Select a predefined vibe"
              >
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest flex items-center gap-2">
                    <Music className="w-3 h-3" />
                    Generation Controls
                  </h3>
                  <button
                    onClick={handleShuffleVibe}
                    disabled={isShuffling || !isCameraActive}
                    className={`flex items-center gap-1.5 px-2 py-0.5 text-[9px] uppercase font-bold border transition-all duration-300 ${
                      isShuffling
                        ? "bg-white/10 border-white/10 text-white/40 cursor-not-allowed"
                        : "bg-white/5 border-white/20 text-white/70 hover:bg-white/20 hover:text-white hover:border-white shadow-[0_0_8px_rgba(255,255,255,0.05)]"
                    } disabled:opacity-40 disabled:cursor-not-allowed`}
                    title={!isCameraActive ? "Start system to shuffle vibe" : "Force AI to regenerate audio profile based on current detected entities"}
                  >
                    {isShuffling ? (
                      <Loader2 className="w-2.5 h-2.5 animate-spin text-white/50" />
                    ) : (
                      <Shuffle className="w-2.5 h-2.5" />
                    )}
                    <span>{isShuffling ? "Shuffling..." : "Shuffle Vibe"}</span>
                  </button>
                </div>

                <div className="mb-4">
                  <div className="flex justify-between text-[10px] font-bold text-white/50 uppercase tracking-widest mb-2">
                    <span>Intensity / Volatility</span>
                    <span className="text-white/80">
                      {intensity.toFixed(1)}x
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="2.0"
                    step="0.1"
                    value={intensity}
                    onChange={(e) => setIntensity(parseFloat(e.target.value))}
                    className="w-full accent-white bg-white/10 h-1 appearance-none cursor-pointer"
                  />
                  <p className="text-[9px] text-white/40 mt-1 uppercase tracking-wider">
                    Affects note frequency and ambient drone density
                  </p>
                </div>

                <div className="mb-4">
                  <div className="flex justify-between text-[10px] font-bold text-white/50 uppercase tracking-widest mb-2">
                    <span>Crossfade Speed</span>
                    <span className="text-white/80">
                      {crossfadeSpeed === 0.02
                        ? "Normal"
                        : crossfadeSpeed > 0.02
                          ? "Snappy"
                          : "Ethereal"}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.005"
                    max="0.1"
                    step="0.005"
                    value={crossfadeSpeed}
                    onChange={(e) =>
                      setCrossfadeSpeed(parseFloat(e.target.value))
                    }
                    className="w-full accent-white bg-white/10 h-1 appearance-none cursor-pointer"
                  />
                  <p className="text-[9px] text-white/40 mt-1 uppercase tracking-wider">
                    Controls transition duration between musical vibes
                  </p>
                </div>

                <div className="mb-4">
                  <div className="flex justify-between text-[10px] font-bold text-white/50 uppercase tracking-widest mb-2">
                    <span>Visualizer Sensitivity</span>
                    <span className="text-white/80">
                      {visualizerSensitivity.toFixed(1)}x
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="3.0"
                    step="0.1"
                    value={visualizerSensitivity}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      setVisualizerSensitivity(val);
                      visualizerSensitivityRef.current = val;
                    }}
                    className="w-full accent-white bg-white/10 h-1 appearance-none cursor-pointer"
                  />
                  <p className="text-[9px] text-white/40 mt-1 uppercase tracking-wider">
                    Affects wireframe and particle reactivity to facial
                    movements
                  </p>
                </div>

                <div className="mb-4">
                  <div className="flex justify-between text-[10px] font-bold text-white/50 uppercase tracking-widest mb-2">
                    <span>Global Tempo (BPM)</span>
                    <span className="text-white/80">
                      {globalBpm === null || globalBpm === 0
                        ? "Auto"
                        : `${globalBpm} BPM`}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="200"
                    step="10"
                    value={globalBpm || 0}
                    onChange={(e) => {
                      const val = parseInt(e.target.value);
                      const newBpm = val === 0 ? null : val;
                      setGlobalBpm(newBpm);
                      globalBpmRef.current = newBpm;
                      if (proceduralEngineRef.current) {
                        proceduralEngineRef.current.setGlobalBpm(newBpm);
                      }
                    }}
                    className="w-full accent-white bg-white/10 h-1 appearance-none cursor-pointer mb-2"
                  />
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-[9px] text-white/40 uppercase tracking-wider flex-1 leading-tight">
                      0 for Auto. Overrides procedural tempo.
                    </p>
                    <button
                      onClick={handleTapTempo}
                      className={`px-2.5 py-1 text-[9px] font-mono font-bold uppercase tracking-wider border rounded-sm transition-all duration-150 flex items-center gap-1 shrink-0 ${
                        isTapping
                          ? "bg-white text-black border-white scale-95 shadow-[0_0_12px_rgba(255,255,255,0.6)]"
                          : "bg-white/5 border-white/20 text-white/80 hover:bg-white/15 hover:text-white hover:border-white/40"
                      }`}
                      title="Tap repeatedly to set custom tempo"
                    >
                      <Activity className={`w-2.5 h-2.5 ${isTapping ? "animate-ping text-black" : "text-white/60 animate-pulse"}`} />
                      Tap Tempo
                    </button>
                  </div>
                </div>

                <div className="mb-4">
                  <div className="flex justify-between text-[10px] font-bold text-white/50 uppercase tracking-widest mb-2">
                    <span>Reverb / Delay Amount</span>
                    <span className="text-white/80">
                      {Math.round(delayAmount * 100)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={delayAmount}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      setDelayAmount(val);
                      if (proceduralEngineRef.current) {
                        proceduralEngineRef.current.setDelayAmount(val);
                      }
                    }}
                    className="w-full accent-white bg-white/10 h-1 appearance-none cursor-pointer"
                  />
                  <p className="text-[9px] text-white/40 mt-1 uppercase tracking-wider">
                    Adds spatial depth and echoes
                  </p>
                </div>

                <div className="mb-4">
                  <div className="flex justify-between text-[10px] font-bold text-white/50 uppercase tracking-widest mb-2">
                    <span>Delay Time</span>
                    <span className="text-white/80">
                      {delayTime.toFixed(2)}s
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="2.0"
                    step="0.01"
                    value={delayTime}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      setDelayTime(val);
                      if (proceduralEngineRef.current) {
                        proceduralEngineRef.current.setDelayTime(val);
                      }
                    }}
                    className="w-full accent-white bg-white/10 h-1 appearance-none cursor-pointer"
                  />
                </div>

                <div className="mb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex flex-col">
                      <span className="text-[10px] font-bold text-white/50 uppercase tracking-widest">
                        Beat Sync (4/4)
                      </span>
                      <span className="text-[9px] text-white/40 uppercase tracking-wider">
                        Quantizes rhythmic styles
                      </span>
                    </div>
                    <button
                      onClick={() => {
                        playHoverSound();
                        setBeatSync(!beatSync);
                      }}
                      className={`w-8 h-4 rounded-full border border-white/30 flex items-center p-0.5 transition-colors ${beatSync ? "bg-white/20" : "bg-transparent"}`}
                    >
                      <div
                        className={`w-3 h-3 bg-white rounded-full transition-transform ${beatSync ? "translate-x-4" : "translate-x-0"}`}
                      />
                    </button>
                  </div>
                </div>

                <div className="mb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex flex-col">
                      <span className="text-[10px] font-bold text-white/50 uppercase tracking-widest">
                        Auto-Gain (Compressor)
                      </span>
                      <span className="text-[9px] text-white/40 uppercase tracking-wider">
                        Prevents volume spikes
                      </span>
                    </div>
                    <button
                      onClick={() => {
                        playHoverSound();
                        const newVal = !autoGain;
                        setAutoGain(newVal);
                        if (proceduralEngineRef.current) {
                          proceduralEngineRef.current.setAutoGain(newVal);
                        }
                      }}
                      className={`w-8 h-4 rounded-full border border-white/30 flex items-center p-0.5 transition-colors ${autoGain ? "bg-white/20" : "bg-transparent"}`}
                    >
                      <div
                        className={`w-3 h-3 bg-white rounded-full transition-transform ${autoGain ? "translate-x-4" : "translate-x-0"}`}
                      />
                    </button>
                  </div>
                </div>

                <div className="mb-4">
                  <div className="flex flex-col">
                    <label className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-1">
                      Force Scale Tonality
                    </label>
                    <select
                      value={forceScale || ""}
                      onChange={(e) => {
                        const val =
                          e.target.value === "" ? null : e.target.value;
                        setForceScale(val);
                        forceScaleRef.current = val;
                        if (proceduralEngineRef.current) {
                          proceduralEngineRef.current.setForceScale(val);
                        }
                      }}
                      className="bg-black/50 border border-white/20 text-white text-[10px] uppercase tracking-wider p-2 w-full appearance-none outline-none cursor-pointer focus:border-white/50"
                    >
                      <option value="">Auto (By Emotion)</option>
                      <option value="major">Major</option>
                      <option value="minor">Minor</option>
                      <option value="pentatonic">Pentatonic</option>
                      <option value="cyberpunk">Cyberpunk</option>
                      <option value="drone">Drone</option>
                      <option value="melancholic">Melancholic</option>
                      <option value="dissonant">Dissonant</option>
                      <option value="tribal">Tribal</option>
                    </select>
                    <p className="text-[9px] text-white/40 mt-1 uppercase tracking-wider">
                      Overrides auto tonality from vibe
                    </p>
                  </div>
                </div>

                <div className="mb-4">
                  <div className="flex justify-between text-[10px] font-bold text-white/50 uppercase tracking-widest mb-2">
                    <span>HUD Theme</span>
                  </div>
                  <div className="grid grid-cols-3 gap-1">
                    {["white", "amber", "cyan"].map((themeName) => (
                      <button
                        key={themeName}
                        onClick={() => {
                          playHoverSound();
                          setHudTheme(themeName as any);
                          hudThemeRef.current = themeName as any;
                        }}
                        onMouseEnter={playHoverSound}
                        className={`text-[9px] uppercase font-bold py-1.5 border transition-colors ${
                          hudTheme === themeName
                            ? "bg-white/20 border-white text-white shadow-[0_0_10px_rgba(255,255,255,0.2)]"
                            : "bg-black/50 border-white/20 text-white/50 hover:bg-white/10 hover:text-white"
                        }`}
                      >
                        {themeName}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="mb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex flex-col">
                      <span className="text-[10px] font-bold text-white/50 uppercase tracking-widest">
                        Ambient Light Mode
                      </span>
                      <span className="text-[9px] text-white/40 uppercase tracking-wider">
                        Adapts background to frame color
                      </span>
                    </div>
                    <button
                      onClick={() => {
                        playHoverSound();
                        setIsAmbientLightMode(!isAmbientLightMode);
                      }}
                      className={`w-8 h-4 rounded-full border border-white/30 flex items-center p-0.5 transition-colors ${isAmbientLightMode ? "bg-white/20" : "bg-transparent"}`}
                    >
                      <div
                        className={`w-3 h-3 bg-white rounded-full transition-transform ${isAmbientLightMode ? "translate-x-4" : "translate-x-0"}`}
                      />
                    </button>
                  </div>
                </div>

                <div className="mb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex flex-col">
                      <span className="text-[10px] font-bold text-white/50 uppercase tracking-widest">
                        Emotion Overlay
                      </span>
                      <span className="text-[9px] text-white/40 uppercase tracking-wider">
                        Displays state above user face
                      </span>
                    </div>
                    <button
                      onClick={() => {
                        playHoverSound();
                        setIsEmotionOverlayEnabled(!isEmotionOverlayEnabled);
                      }}
                      className={`w-8 h-4 rounded-full border border-white/30 flex items-center p-0.5 transition-colors ${isEmotionOverlayEnabled ? "bg-white/20" : "bg-transparent"}`}
                    >
                      <div
                        className={`w-3 h-3 bg-white rounded-full transition-transform ${isEmotionOverlayEnabled ? "translate-x-4" : "translate-x-0"}`}
                      />
                    </button>
                  </div>
                </div>

                <div className="mb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex flex-col">
                      <span className="text-[10px] font-bold text-white/50 uppercase tracking-widest">
                        Tactile Haptics
                      </span>
                      <span className="text-[9px] text-white/40 uppercase tracking-wider">
                        BPM-synced physical heartbeat
                      </span>
                    </div>
                    <button
                      onClick={() => {
                        playHoverSound();
                        const nextVal = !isHapticFeedbackEnabled;
                        setIsHapticFeedbackEnabled(nextVal);
                        if (nextVal && typeof navigator !== "undefined" && navigator.vibrate) {
                          // Quick crisp triple acknowledgment buzz scaled by current intensity
                          const intensityMultiplier = vibrationIntensityRef.current / 100;
                          navigator.vibrate([
                            Math.round(30 * intensityMultiplier),
                            Math.round(40 * intensityMultiplier),
                            Math.round(30 * intensityMultiplier),
                            Math.round(40 * intensityMultiplier),
                            Math.round(50 * intensityMultiplier)
                          ].filter(v => v > 0));
                        }
                      }}
                      className={`w-8 h-4 rounded-full border border-white/30 flex items-center p-0.5 transition-colors ${isHapticFeedbackEnabled ? "bg-white/20" : "bg-transparent"}`}
                    >
                      <div
                        className={`w-3 h-3 bg-white rounded-full transition-transform ${isHapticFeedbackEnabled ? "translate-x-4" : "translate-x-0"}`}
                      />
                    </button>
                  </div>

                  {isHapticFeedbackEnabled && (
                    <div className="mt-3 pl-2 border-l border-white/10">
                      <div className="flex justify-between text-[9px] font-bold text-white/40 uppercase tracking-widest mb-1.5">
                        <span>Vibration Intensity</span>
                        <span className="text-white/70 font-mono">{vibrationIntensity}%</span>
                      </div>
                      <input
                        type="range"
                        min="10"
                        max="200"
                        step="10"
                        value={vibrationIntensity}
                        onChange={(e) => {
                          const val = parseInt(e.target.value);
                          setVibrationIntensity(val);
                          vibrationIntensityRef.current = val;
                          if (typeof navigator !== "undefined" && navigator.vibrate) {
                            // Quick single preview pulse matching the updated intensity
                            navigator.vibrate(Math.max(5, Math.round(25 * (val / 100))));
                          }
                        }}
                        className="w-full accent-white bg-white/10 h-1 appearance-none cursor-pointer"
                      />
                      <p className="text-[8px] text-white/30 mt-1 uppercase tracking-wider">
                        Scales the physical duration of heartbeat pulses
                      </p>
                    </div>
                  )}
                </div>

                <div className="mb-2">
                  <div className="flex justify-between text-[10px] font-bold text-white/50 uppercase tracking-widest mb-2">
                    <span>Visualizer Style</span>
                  </div>
                  <div className="grid grid-cols-3 gap-1">
                    {["default", "wireframe", "particle"].map((mode) => (
                      <button
                        key={mode}
                        onClick={() => {
                          playHoverSound();
                          setVisualizerMode(mode as any);
                          visualizerModeRef.current = mode as any;
                        }}
                        onMouseEnter={playHoverSound}
                        className={`text-[9px] uppercase font-bold py-1.5 border transition-colors ${
                          visualizerMode === mode
                            ? "bg-white/20 border-white text-white shadow-[0_0_10px_rgba(255,255,255,0.2)]"
                            : "bg-black/50 border-white/20 text-white/50 hover:bg-white/10 hover:text-white"
                        }`}
                      >
                        {mode === "default" ? "Scan" : mode}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="mb-4 pt-4 border-t border-white/10">
                  <div className="flex items-center justify-between">
                    <div className="flex flex-col">
                      <span className="text-[10px] font-bold text-white/50 uppercase tracking-widest flex items-center gap-2">
                        <Disc
                          className={`w-3 h-3 ${isRecording ? "text-red-500 animate-pulse" : ""}`}
                        />
                        Audio Export
                      </span>
                      <span className="text-[9px] text-white/40 uppercase tracking-wider">
                        Record & download mix
                      </span>
                    </div>
                    <button
                      onClick={async () => {
                        playHoverSound();
                        if (isRecording) {
                          setIsRecording(false);
                          if (proceduralEngineRef.current) {
                            const url =
                              await proceduralEngineRef.current.stopRecording();
                            if (url) {
                              const a = document.createElement("a");
                              a.href = url;
                              a.download = `serene-export-${Date.now()}.webm`;
                              document.body.appendChild(a);
                              a.click();
                              document.body.removeChild(a);
                            }
                          }
                        } else {
                          setIsRecording(true);
                          if (proceduralEngineRef.current) {
                            proceduralEngineRef.current.startRecording();
                          }
                        }
                      }}
                      className={`px-3 py-1.5 rounded-full border text-[9px] font-bold uppercase tracking-wider transition-colors flex items-center gap-1 ${
                        isRecording
                          ? "border-red-500 bg-red-500/20 text-red-500"
                          : "border-white/30 bg-transparent text-white/70 hover:bg-white/10 hover:text-white"
                      }`}
                    >
                      {isRecording ? (
                        <>
                          <Square className="w-3 h-3 fill-current" />
                          Stop
                        </>
                      ) : (
                        <>
                          <Download className="w-3 h-3" />
                          Record
                        </>
                      )}
                    </button>
                  </div>
                </div>

                <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-3 flex items-center gap-2 mt-4 border-t border-white/10 pt-4">
                  <Music className="w-3 h-3" />
                  Override Vibe
                </h3>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { label: "Auto-Detect", value: null },
                    {
                      label: "Deep Space",
                      value: "deep space ambient drone, sci-fi",
                    },
                    {
                      label: "Rainy Cafe",
                      value: "rainy coffee shop jazz, chill acoustic",
                    },
                    {
                      label: "Glitch Core",
                      value: "cyberpunk glitch hop, electronic",
                    },
                  ].map((preset) => (
                    <button
                      key={preset.label}
                      onClick={() => {
                        playHoverSound();
                        handlePresetSelect(preset.label, preset.value);
                      }}
                      onMouseEnter={playHoverSound}
                      className={`text-[10px] uppercase font-bold py-2 border transition-colors ${
                        presetVibe === preset.value
                          ? "bg-white/20 border-white text-white shadow-[0_0_10px_rgba(255,255,255,0.2)]"
                          : "bg-black/50 border-white/20 text-white/50 hover:bg-white/10 hover:text-white"
                      }`}
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Audio Profile */}
              <div
                className="order-3 lg:order-2 w-full bg-black/40 backdrop-blur-md border border-white/20 p-5 shadow-[0_0_30px_rgba(0,0,0,0.8)]"
                title="AI-generated musical prompt based on your environment and mood"
              >
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest">
                    Audio Profile
                  </h3>
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => {
                        playHoverSound();
                        const nextLocked = !isPromptLocked;
                        setIsPromptLocked(nextLocked);
                        isPromptLockedRef.current = nextLocked;
                      }}
                      className={`flex items-center gap-1.5 text-[10px] transition-colors ${isPromptLocked ? "text-amber-400 hover:text-amber-300" : "text-white/50 hover:text-white"}`}
                      title={
                        isPromptLocked
                          ? "Unlock Audio Profile"
                          : "Lock Audio Profile (freeze updates)"
                      }
                    >
                      {isPromptLocked ? (
                        <Lock className="w-3 h-3 fill-current" />
                      ) : (
                        <Unlock className="w-3 h-3" />
                      )}
                      <span className="uppercase">
                        {isPromptLocked ? "Locked" : "Lock"}
                      </span>
                    </button>
                    <button
                      onClick={captureSnapshot}
                      className="flex items-center gap-1.5 text-[10px] text-white/50 hover:text-white transition-colors"
                    >
                      <Camera className="w-3 h-3" />
                      <span className="uppercase">Save Snapshot</span>
                    </button>
                    <button
                      onClick={copyVibeToClipboard}
                      className="flex items-center gap-1.5 text-[10px] text-white/50 hover:text-white transition-colors"
                    >
                      {isCopied ? (
                        <Check className="w-3 h-3 text-green-400" />
                      ) : (
                        <Copy className="w-3 h-3" />
                      )}
                      <span className="uppercase">
                        {isCopied ? "Copied" : "Copy Vibe"}
                      </span>
                    </button>
                    <button
                      onClick={copyShareLinkToClipboard}
                      className="flex items-center gap-1.5 text-[10px] text-white/50 hover:text-white transition-colors"
                      title="Copy deep link with all parameters"
                    >
                      {isLinkCopied ? (
                        <Check className="w-3 h-3 text-green-400" />
                      ) : (
                        <Link className="w-3 h-3" />
                      )}
                      <span className="uppercase">
                        {isLinkCopied ? "Copied Link" : "Copy Link"}
                      </span>
                    </button>
                  </div>
                </div>
                <div className="relative overflow-hidden">
                  <div className="absolute top-0 left-0 w-0.5 h-full bg-white shadow-[0_0_10px_rgba(255,255,255,0.8)]" />
                  <p className="text-xs leading-relaxed text-white/90 pl-3">
                    {currentPrompt}
                  </p>
                </div>
              </div>

              {/* Favorites Sidebar section */}
              {favoriteVibes.length > 0 && (
                <div
                  className="order-4 lg:order-3 w-full bg-black/40 backdrop-blur-md border border-red-500/30 p-5 shadow-[0_0_30px_rgba(239,68,68,0.2)]"
                  title="Saved favorite audio profiles"
                >
                  <h3 className="text-[10px] font-bold text-red-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                    <Heart className="w-3 h-3 fill-current" />
                    Favorites
                  </h3>
                  <ul className="space-y-3">
                    <AnimatePresence>
                      {favoriteVibes.map((vibe, idx) => (
                        <motion.li
                          key={vibe + idx}
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={{ opacity: 0, height: 0 }}
                          className="relative overflow-hidden group flex items-start gap-2 cursor-pointer"
                          onClick={() => {
                            playHoverSound();
                            setCurrentPrompt(vibe);
                            if (proceduralEngineRef.current) {
                              proceduralEngineRef.current.setVibe(vibe);
                            }
                          }}
                        >
                          <div className="absolute top-0 left-0 w-0.5 h-full bg-red-400/60" />
                          <p className="text-[10px] leading-relaxed text-red-100/70 pl-3 line-clamp-2 flex-1 group-hover:text-red-100 transition-colors">
                            {vibe}
                          </p>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              playHoverSound();
                              setFavoriteVibes((prev) =>
                                prev.filter((v) => v !== vibe),
                              );
                            }}
                            className="p-1 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity text-red-500 hover:text-red-400"
                            title="Remove from Favorites"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </motion.li>
                      ))}
                    </AnimatePresence>
                  </ul>
                </div>
              )}

              {/* Activity History Chart */}
              <div className="order-5 lg:order-4 w-full">
                <ActivityChart
                  data={chartData}
                  themeRgb={themeRgbMap[hudTheme]}
                />
              </div>

              {/* Recent Profiles List */}
              {recentVibes.length > 0 && (
                <div
                  className="order-6 lg:order-5 w-full bg-black/40 backdrop-blur-md border border-white/20 p-5 shadow-[0_0_30px_rgba(0,0,0,0.8)]"
                  title="Recently generated audio profiles"
                >
                  <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-3 flex items-center gap-2">
                    <History className="w-3 h-3" />
                    Recent Profiles
                  </h3>
                  <div className="max-h-[150px] overflow-y-auto pr-2 custom-scrollbar">
                    <ul className="space-y-3">
                      <AnimatePresence>
                        {recentVibes.map((vibe, idx) => (
                          <motion.li
                            key={vibe + idx}
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: "auto" }}
                            exit={{ opacity: 0, height: 0 }}
                            className="relative overflow-hidden group flex items-start gap-2 cursor-pointer"
                            onClick={() => {
                              playHoverSound();
                              setCurrentPrompt(vibe);
                              if (proceduralEngineRef.current) {
                                proceduralEngineRef.current.setVibe(vibe);
                              }
                            }}
                          >
                            <div className="absolute top-0 left-0 w-0.5 h-full bg-white/40 group-hover:bg-white transition-colors" />
                            <p className="text-[10px] leading-relaxed text-white/60 pl-3 line-clamp-2 flex-1 group-hover:text-white transition-colors">
                              {vibe}
                            </p>
                          </motion.li>
                        ))}
                      </AnimatePresence>
                    </ul>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Snapshot Gallery & Timelapse */}
        <div className="w-full mt-4 bg-black/40 backdrop-blur-md border border-white/20 p-5 shadow-[0_0_30px_rgba(0,0,0,0.8)] pointer-events-auto shrink-0">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-4 pb-3 border-b border-white/10">
            <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest flex items-center gap-2">
              <Camera className="w-3.5 h-3.5" />
              Snapshot Gallery & Timelapse
            </h3>

            {/* Auto-Capture Timelapse Controls */}
            <div className="flex flex-wrap items-center gap-4 bg-white/5 border border-white/10 px-3 py-1.5 rounded-sm">
              <div className="flex items-center gap-2">
                <span className="text-[9px] font-bold text-white/40 uppercase tracking-widest">Timelapse Capture</span>
                <button
                  onClick={() => {
                    playHoverSound();
                    setIsAutoCaptureActive(!isAutoCaptureActive);
                  }}
                  className={`w-8 h-4 rounded-full border border-white/30 flex items-center p-0.5 transition-colors ${
                    isAutoCaptureActive ? "bg-white/20" : "bg-transparent"
                  }`}
                  title={!isCameraActive ? "Activate system camera to start timelapse" : "Toggle auto-capturing snapshots at regular intervals"}
                >
                  <div
                    className={`w-2.5 h-2.5 rounded-full bg-white transition-transform duration-200 ${
                      isAutoCaptureActive ? "translate-x-4 bg-white" : "translate-x-0 bg-white/40"
                    }`}
                  />
                </button>
              </div>

              {isAutoCaptureActive && (
                <div className="flex items-center gap-2">
                  <span className="text-[9px] font-mono text-white/70 bg-white/10 px-1.5 py-0.5 rounded-sm flex items-center gap-1">
                    <Timer className="w-2.5 h-2.5 animate-spin" />
                    Next in: {timeUntilNextCapture}s
                  </span>
                  <div className="w-16 bg-white/10 h-1.5 rounded-full overflow-hidden">
                    <div
                      className="bg-white h-full transition-all duration-1000 ease-linear"
                      style={{ width: `${(timeUntilNextCapture / autoCaptureInterval) * 100}%` }}
                    />
                  </div>
                </div>
              )}

              <div className="flex items-center gap-2">
                <span className="text-[9px] font-bold text-white/40 uppercase tracking-widest">Interval</span>
                <input
                  type="range"
                  min="10"
                  max="120"
                  step="5"
                  value={autoCaptureInterval}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    setAutoCaptureInterval(val);
                    setTimeUntilNextCapture(val);
                  }}
                  className="w-20 accent-white bg-white/10 h-1 appearance-none cursor-pointer"
                  title="Adjust auto-capture interval"
                />
                <span className="text-[9px] font-mono font-bold text-white/70 min-w-[28px]">
                  {autoCaptureInterval}s
                </span>
              </div>
            </div>
          </div>

          {snapshots.length > 0 ? (
            <div className="flex gap-4 overflow-x-auto pb-2 snap-x custom-scrollbar">
              {snapshots.map((snap) => (
                <div
                  key={snap.id}
                  onClick={() => recallSnapshot(snap)}
                  className="min-w-[200px] w-[200px] shrink-0 border border-white/10 snap-center group relative overflow-hidden bg-black cursor-pointer"
                  title="Click to recall this vibe and parameters"
                >
                  <div className="relative">
                    <img
                      src={snap.imageUri}
                      alt="Snapshot"
                      className="w-full h-auto object-cover grayscale contrast-125 transition-transform duration-500 group-hover:scale-105"
                    />
                    {snap.isAutoCaptured && (
                      <span className="absolute top-2 left-2 px-1.5 py-0.5 bg-black/75 border border-white/20 text-[7px] font-mono font-bold text-amber-400 uppercase tracking-widest">
                        Timelapse
                      </span>
                    )}
                  </div>
                  <div className="absolute inset-0 bg-gradient-to-t from-black/95 via-black/40 to-black/90 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-between p-3">
                    <div className="flex justify-between items-start w-full">
                      <p className="text-[9px] text-white/60 font-mono">
                        {new Date(snap.timestamp).toLocaleTimeString()}
                      </p>
                      <div className="flex flex-col gap-1.5 items-end">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            recallSnapshot(snap);
                          }}
                          className="text-[9px] font-mono font-bold uppercase tracking-wider bg-white/10 hover:bg-white/25 text-white px-2 py-1 border border-white/20 rounded-sm transition-colors"
                        >
                          Recall
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            playHoverSound();
                            handleGenerateQR(snap);
                          }}
                          className="text-[9px] font-mono font-bold uppercase tracking-wider bg-white/10 hover:bg-white/25 text-white px-2 py-1 border border-white/20 rounded-sm transition-colors flex items-center gap-1"
                          title="Generate QR code for this audio profile"
                        >
                          <QrCode className="w-2.5 h-2.5" />
                          Share QR
                        </button>
                      </div>
                    </div>
                    <p className="text-[10px] text-white/90 leading-tight line-clamp-3 mt-auto font-mono">
                      {snap.audioProfile}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="w-full py-10 flex flex-col items-center justify-center text-center bg-white/5 border border-dashed border-white/10 rounded-sm">
              <Camera className="w-8 h-8 text-white/20 mb-2 animate-pulse" />
              <p className="text-[11px] text-white/60 uppercase tracking-wider font-mono">No snapshots captured yet</p>
              <p className="text-[9px] text-white/40 max-w-sm mt-1 uppercase tracking-wider leading-relaxed">
                {!isCameraActive 
                  ? "Start the system session to take manual snapshots or enable the timelapse capture timer." 
                  : "Turn on Timelapse Capture or click 'Save Snapshot' above to record your session moods."}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* QR Code Modal */}
      <AnimatePresence>
        {selectedQrSnapshot && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-sm pointer-events-auto"
            onClick={() => setSelectedQrSnapshot(null)}
          >
            <motion.div
              initial={{ scale: 0.95, y: 15, opacity: 0 }}
              animate={{ scale: 1, y: 0, opacity: 1 }}
              exit={{ scale: 0.95, y: 15, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-zinc-950 border border-white/25 p-6 max-w-sm w-full shadow-[0_0_50px_rgba(255,255,255,0.1)] relative overflow-hidden flex flex-col items-center"
            >
              {/* Abstract decorative tech grid line */}
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-white/50 to-transparent" />
              
              <div className="flex justify-between items-center w-full mb-4 pb-2 border-b border-white/10">
                <h3 className="text-xs font-mono font-bold text-white uppercase tracking-widest flex items-center gap-2">
                  <QrCode className="w-4 h-4 text-white" />
                  Share Vibe Profile
                </h3>
                <button
                  onClick={() => setSelectedQrSnapshot(null)}
                  className="p-1 border border-white/10 bg-black/50 hover:bg-white/10 text-white/50 hover:text-white transition-colors"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>

              {/* Snapshot thumbnail preview */}
              <div className="w-full flex items-center gap-3 bg-white/5 border border-white/10 p-2 mb-4 rounded-sm">
                <img
                  src={selectedQrSnapshot.imageUri}
                  alt="Snapshot"
                  className="w-12 h-12 object-cover grayscale contrast-125 border border-white/15"
                />
                <div className="flex-1 overflow-hidden">
                  <p className="text-[9px] font-mono text-white/50 uppercase tracking-widest">
                    Captured: {new Date(selectedQrSnapshot.timestamp).toLocaleTimeString()}
                  </p>
                  <p className="text-[10px] text-white/85 line-clamp-2 italic font-mono leading-tight mt-0.5">
                    "{selectedQrSnapshot.audioProfile}"
                  </p>
                </div>
              </div>

              {/* QR Code Container */}
              <div className="relative p-4 bg-white rounded-md shadow-inner flex items-center justify-center mb-4 border-2 border-white/20">
                {qrDataUrl ? (
                  <img
                    src={qrDataUrl}
                    alt="Vibe QR Code"
                    className="w-44 h-44"
                  />
                ) : (
                  <div className="w-44 h-44 flex items-center justify-center">
                    <Loader2 className="w-8 h-8 text-black animate-spin" />
                  </div>
                )}
              </div>

              <p className="text-[10px] text-center text-white/40 font-mono uppercase tracking-wider mb-5">
                Scan with any device to read this audio profile
              </p>

              {/* Action Buttons */}
              <div className="grid grid-cols-2 gap-2 w-full">
                <button
                  onClick={() => {
                    playHoverSound();
                    navigator.clipboard.writeText(selectedQrSnapshot.audioProfile);
                    setIsQrTextCopied(true);
                    setTimeout(() => setIsQrTextCopied(false), 2000);
                  }}
                  className="py-2.5 px-3 bg-white/5 hover:bg-white/10 border border-white/20 text-white/70 hover:text-white text-[10px] font-mono font-bold uppercase tracking-widest transition-colors flex items-center justify-center gap-1.5"
                >
                  {isQrTextCopied ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-green-400" />
                      Copied
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      Copy Text
                    </>
                  )}
                </button>
                <a
                  href={qrDataUrl}
                  download={`vibe-qr-${selectedQrSnapshot.id}.png`}
                  onClick={playHoverSound}
                  className="py-2.5 px-3 bg-white hover:bg-white/90 text-black text-[10px] font-mono font-bold uppercase tracking-widest transition-colors flex items-center justify-center gap-1.5 text-center"
                >
                  <Download className="w-3.5 h-3.5" />
                  Download
                </a>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error Modal */}
      <AnimatePresence>
        {errorMsg && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm pointer-events-auto"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-zinc-900 border border-red-500/50 p-6 max-w-md w-full shadow-[0_0_40px_rgba(239,68,68,0.2)] relative"
            >
              <div className="flex items-start gap-4 mb-6">
                <div className="p-3 bg-red-500/10 border border-red-500/30 shrink-0">
                  <AlertCircle className="w-6 h-6 text-red-500" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-red-500 uppercase tracking-widest">
                    {status}
                  </h3>
                  <p className="text-sm mt-2 text-red-400/80 leading-relaxed">
                    {errorMsg}
                  </p>
                </div>
              </div>

              <button
                onClick={() => setErrorMsg(null)}
                className={`w-full py-3 text-xs font-mono font-bold uppercase tracking-widest transition-colors ${
                  status === "Camera Error"
                    ? "bg-red-500/20 hover:bg-red-500/30 border border-red-500/50 text-red-400"
                    : "bg-white/5 hover:bg-white/10 border border-white/20 text-white/70"
                }`}
              >
                Dismiss
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Info Toast (Center Bottom) */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex flex-col justify-end items-center pointer-events-none z-30 w-[calc(100%-2rem)] sm:w-full max-w-md">
        <AnimatePresence>
          {infoMsg && !errorMsg && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className="bg-black/80 backdrop-blur-md border border-white/30 p-4 flex items-start gap-3 text-white shadow-[0_0_20px_rgba(255,255,255,0.1)] mb-4 w-full"
            >
              <Music className="w-5 h-5 shrink-0 mt-0.5" />
              <div>
                <h3 className="font-bold text-sm">{status}</h3>
                <p className="text-xs mt-1 text-white/80">{infoMsg}</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Info Modal */}
      <AnimatePresence>
        {isInfoOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm pointer-events-auto"
            onClick={() => setIsInfoOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-zinc-900 border border-white/20 p-6 max-w-lg w-full shadow-[0_0_40px_rgba(0,0,0,0.8)] relative max-h-[90vh] overflow-y-auto"
            >
              <div className="flex flex-col-reverse sm:flex-row sm:items-start justify-between gap-4 mb-4">
                <h2 className="text-xl font-bold text-white flex items-center gap-2 self-start">
                  <Info className="w-5 h-5 shrink-0" />
                  About Vision to Music
                </h2>
                <button
                  onClick={() => setIsInfoOpen(false)}
                  className="p-2 shrink-0 border border-white/20 bg-black/50 hover:bg-white/10 text-white/50 hover:text-white transition-colors self-end sm:self-auto"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="space-y-4 text-sm text-white/80 leading-relaxed">
                <p>
                  <strong>Vision to Music</strong> uses your device's camera to
                  analyze your facial expressions and the objects around you in
                  real-time.
                </p>
                <p>
                  Based on this visual data, it generates a continuous,
                  procedural ambient soundscape that matches your mood and
                  environment.
                </p>
                <ul className="list-disc pl-5 space-y-2 text-white/70">
                  <li>
                    <strong>Biometric Scan:</strong> Tracks your facial
                    landmarks to determine your current emotion (happy, sad,
                    surprised, angry, fear, disgust).
                  </li>
                  <li>
                    <strong>Entities:</strong> Detects objects in your
                    environment (like laptops, cups, plants) to influence the
                    musical vibe.
                  </li>
                  <li>
                    <strong>Audio Profile:</strong> The AI generates a
                    descriptive prompt based on the scene, which drives the
                    procedural music engine.
                  </li>
                </ul>
                <p className="text-xs text-white/50 mt-4 pt-4 border-t border-white/10">
                  Note: All processing happens locally in your browser or via
                  secure API calls. No video data is saved or transmitted.
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
