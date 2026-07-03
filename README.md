<div align="center">

<img src="https://ai.google.dev/static/site-assets/images/share-ais-513315318.png" alt="VibeFlow — see the world, hear the vibe" width="100%" />

<br /><br />

<a href="https://vibeflow-three.vercel.app"><img src="https://img.shields.io/badge/Live-vibeflow--three.vercel.app-000000?style=for-the-badge&logo=vercel&logoColor=white" alt="live on vercel" /></a>
<img src="https://img.shields.io/badge/VibeFlow-Vision_Sync-6C5CE7?style=for-the-badge&logoColor=white" alt="app" />
<img src="https://img.shields.io/badge/Built_in-AI_Studio-4285F4?style=for-the-badge&logo=googlegemini&logoColor=white" alt="ai studio" />
<img src="https://img.shields.io/badge/Runs-On--device_Vision-3DDC84?style=for-the-badge&logo=tensorflow&logoColor=white" alt="on-device" />
<img src="https://img.shields.io/badge/Architecture-Local--first-10B981?style=for-the-badge" alt="local-first" />

<br />

<img src="https://img.shields.io/badge/TypeScript-3178c6?style=for-the-badge&logo=typescript&logoColor=white" alt="typescript" />
<img src="https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="react" />
<img src="https://img.shields.io/badge/Vite-6-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="vite" />
<img src="https://img.shields.io/badge/Tailwind_CSS-v4-38BDF8?style=for-the-badge&logo=tailwindcss&logoColor=white" alt="tailwind" />
<img src="https://img.shields.io/badge/Gemini-flash--lite-8E75FF?style=for-the-badge&logo=googlegemini&logoColor=white" alt="gemini" />
<img src="https://img.shields.io/badge/TensorFlow.js-COCO--SSD-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="tfjs" />
<img src="https://img.shields.io/badge/MediaPipe-FaceLandmarker-00A98F?style=for-the-badge&logo=google&logoColor=white" alt="mediapipe" />
<img src="https://img.shields.io/badge/Three.js-Reactive_Visualizer-000000?style=for-the-badge&logo=threedotjs&logoColor=white" alt="three" />
<img src="https://img.shields.io/badge/Web_Audio-Procedural_Engine-F7DF1E?style=for-the-badge&logo=webaudio&logoColor=black" alt="web audio" />

<br /><br />

```text
██╗   ██╗██╗██████╗ ███████╗███████╗██╗      ██████╗ ██╗    ██╗
██║   ██║██║██╔══██╗██╔════╝██╔════╝██║     ██╔═══██╗██║    ██║
██║   ██║██║██████╔╝█████╗  █████╗  ██║     ██║   ██║██║ █╗ ██║
╚██╗ ██╔╝██║██╔══██╗██╔══╝  ██╔══╝  ██║     ██║   ██║██║███╗██║
 ╚████╔╝ ██║██████╔╝███████╗██║     ███████╗╚██████╔╝╚███╔███╔╝
  ╚═══╝  ╚═╝╚═════╝ ╚══════╝╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝ 
```

### **Point your camera. Hear the room.**
### VibeFlow watches what's in front of you — objects, faces, mood — and generates a living soundscape in real time.

</div>

---

## ✦ What is VibeFlow?

**VibeFlow** (aka **Vision Sync**) is a real-time, browser-based **computer-vision instrument**. Aim your webcam at a scene and the app detects the objects in view, reads the facial expression of anyone in frame, distills that into a **3–5 word ambient soundscape** with Gemini, then drives a **procedural Web Audio engine** that plays generative music matching the mood — all reacting live to what the camera sees.

Vision runs **entirely on your device**: object detection (TensorFlow.js **COCO-SSD**) and facial landmarks + emotion (**MediaPipe FaceLandmarker** blendshapes) never leave the browser. The **only** thing sent to the cloud is a short text description of the scene, used to name the vibe. No video, no images, no audio are uploaded.

Built as a **React 19 + Vite 6 + TypeScript** app, styled with **Tailwind CSS v4** and **Motion**, with a **Three.js** reactive visualizer and a raw **Web Audio API** synthesis engine — dark, neon, control-room aesthetic.

<div align="center">

### 🌐 Live on Vercel — **[vibeflow-three.vercel.app](https://vibeflow-three.vercel.app)**
### 🎛️ Also in Google AI Studio — [open the app](https://ai.studio/apps/4fef5f0d-fb0a-41d8-ae14-9f1907d13d22)

</div>

> **Local-first by design.** Snapshots, favorites and settings persist in `localStorage`. Camera frames are processed on-device; only a text scene summary is sent to Gemini. No ads, no tracking, no analytics SDKs.

---

## ⚡ Features

| Feature | Description |
|---------|-------------|
| 👁️ **Real-time Object Detection** | TensorFlow.js **COCO-SSD** labels everything in the camera view, frame by frame — objects become the raw material for the vibe |
| 🙂 **Face & Emotion Tracking** | **MediaPipe FaceLandmarker** draws live facial landmarks and infers emotional state from blendshapes (happy / sad / angry / neutral / fear …) |
| 🤖 **AI Soundscape Naming** | Gemini `flash-lite` turns *scene + mood* into a short ambient descriptor (e.g. `cyberpunk electronic drone`, `melancholy acoustic ambient`) — never "pop", never "upbeat" |
| 🎹 **Procedural Music Engine** | Hand-built **Web Audio** synth: 7 mood scales (major, minor, pentatonic, cyberpunk, drone, melancholic, dissonant, tribal), delay/space, master compressor, live analyser |
| 🌀 **Three.js Visualizer** | Reactive 3D visuals driven by the audio analyser, with adjustable sensitivity |
| 📊 **Activity Chart** | Recharts timeline of detections / intensity over the session |
| 🎚️ **Deep Vibe Controls** | Intensity, beat-sync, auto-gain, crossfade speed, force-scale, global BPM, visualizer sensitivity — all live |
| 🥁 **Tap Tempo** | Tap to set a custom global BPM; lock the groove or let the scene drive it |
| 📸 **Snapshots & Timelapse** | Save the current mood as a snapshot (image + full audio profile + params), or auto-capture at intervals to record a session's emotional arc |
| 🔗 **QR Deep-Link Sharing** | Encode any audio profile + all parameters into a shareable deep link and QR code — recall the exact vibe on another device |
| ❤️ **Favorites & Recents** | Star audio profiles you love; browse recently generated ones |
| ♻️ **Recall & Re-vibe** | Tap any saved snapshot to instantly restore its profile and parameters |
| 🌐 **On-device Privacy** | COCO-SSD + MediaPipe WASM run locally from `/public`; only text scene summaries touch the network |
| 💾 **Local Persistence** | Snapshots, favorites and settings survive reloads via `localStorage` |

---

## 🛠️ Tech Stack

```
Framework       →  React 19 · Vite 6 · TypeScript 5.8
Styling         →  Tailwind CSS v4 (@tailwindcss/vite) · Motion · lucide-react
Vision          →  TensorFlow.js + COCO-SSD (object detection)
                   @mediapipe/tasks-vision · FaceLandmarker (landmarks + emotion blendshapes)
AI              →  @google/genai · Gemini flash-lite (scene → soundscape descriptor)
Audio           →  Raw Web Audio API procedural engine · Tone.js (UI/hover synth)
3D / Charts     →  three.js (reactive visualizer) · Recharts (activity chart)
Sharing         →  qrcode (deep-link QR codes)
Persistence     →  localStorage (snapshots · favorites · settings)
Server          →  Express static host (server.ts) · better-sqlite3 available
Camera          →  getUserMedia (frame permission: camera)
```

> **Vision models are self-hosted.** COCO-SSD, the MediaPipe WASM runtime and the face model ship under `/public` (`/models`, `/wasm`, `/coco`) so detection works without third-party model CDNs.

---

## 🧠 How It Works

```
Camera frame
   │
   ├─▶ COCO-SSD  ───────────▶  detected objects  ┐
   │   (TensorFlow.js)                            │
   │                                              ├─▶  scene summary (text)
   ├─▶ FaceLandmarker ──────▶  emotion + mesh  ───┘        │
   │   (MediaPipe blendshapes)                             │
   │                                                       ▼
   │                                          Gemini flash-lite
   │                                       "3–5 word ambient vibe"
   │                                                       │
   │                                                       ▼
   └─▶ audio analyser ◀──── Procedural Music Engine ◀── mood scale
                │              (Web Audio · 7 scales)
                ▼
        Three.js visualizer  +  Recharts activity
```

Everything left of the "scene summary" arrow stays on-device. Only that short text string is sent to Gemini to name the vibe; the audio itself is synthesized locally in the browser.

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** `>= 18`
- A **Gemini API key** ([AI Studio](https://ai.studio) → API keys)
- A webcam + a browser that supports `getUserMedia` and Web Audio (Chrome recommended)

### Install & run

```bash
# Clone
git clone <your-repo-url> VibeFlow
cd VibeFlow

# Install
npm install

# Configure the AI key
cp .env.example .env.local
# then set GEMINI_API_KEY in .env.local

# Dev server (Vite, port 3000)
npm run dev            # http://localhost:3000

# Type check
npm run lint           # tsc --noEmit

# Production build
npm run build          # → /dist

# Serve the build (Express)
npm start              # server.ts, static host on PORT (default 3000)
```

On first run the browser asks for **camera** permission. Grant it, then press the system/start control in the UI to begin detection and audio.

---

## 🌐 Deployment (Vercel)

VibeFlow is deployed as a static Vite build on **Vercel**.

| Item | Value |
|------|-------|
| **Live URL** | https://vibeflow-three.vercel.app |
| **Project** | `vibeflow` (team `kutluhans-projects-93876a9e`) |
| **Framework** | Vite — `vercel.json` pins `buildCommand: npm run build`, `outputDirectory: dist` |
| **Git** | Connected to the GitHub repo — push to trigger a redeploy |

Deploy from the CLI:

```bash
vercel deploy --prod --yes --scope kutluhans-projects-93876a9e
```

> **Heads-up on the AI key.** `GEMINI_API_KEY` is inlined into the client bundle at build time (`vite define`). The current live deploy ships **without** the key, so AI soundscape naming falls back to the built-in vibe map. To enable Gemini naming, add `GEMINI_API_KEY` in the Vercel project's Environment Variables and redeploy — but note the key becomes visible in the public bundle. For a locked-down setup, proxy Gemini through a serverless function instead of inlining the key.

---

## 🔒 Privacy

| Layer | Implementation |
|-------|----------------|
| **Video / images** | Never uploaded. Object detection and face landmarks run on-device via TensorFlow.js + MediaPipe WASM. |
| **What leaves the device** | Only a short **text** scene summary (object list + emotion word), sent to Gemini to name the soundscape. |
| **Audio** | Fully synthesized in-browser with the Web Audio API — no audio is recorded or sent anywhere. |
| **Storage** | Snapshots, favorites and settings live in `localStorage` on your machine. |
| **Secrets** | `GEMINI_API_KEY` lives in `.env.local` (gitignored). AI Studio injects it at runtime; no key is committed. |
| **Sharing** | QR deep links encode audio parameters only — no personal data. |

---

## 📁 Project Structure

```
VibeFlow/
├── src/
│   ├── App.tsx                        # Main app: camera loop, detection, UI, controls
│   ├── main.tsx                       # React entry
│   ├── types.ts                       # Snapshot model
│   ├── components/
│   │   ├── ThreeVisualizer.tsx        # Audio-reactive three.js visuals
│   │   └── ActivityChart.tsx          # Recharts session timeline
│   └── engine/
│       └── ProceduralMusicEngine.ts   # Web Audio synthesis · 7 mood scales
├── public/
│   └── models/  ·  wasm/  ·  coco/     # Self-hosted MediaPipe + COCO-SSD assets
├── server.ts                          # Express static host for /dist
├── vite.config.ts  ·  tsconfig.json
└── .env.example                       # GEMINI_API_KEY · APP_URL
```

---

## 📄 License

No license file is included yet. Add a `LICENSE` (e.g. MIT) before publishing if you intend others to reuse the code.

---

<div align="center">

Built with React 19 · TensorFlow.js · MediaPipe · Gemini · Web Audio — vision in, vibe out.

*If VibeFlow makes your room sing, give the repo a ⭐*

</div>
