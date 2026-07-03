export class ProceduralMusicEngine {
  audioContext: AudioContext;
  isPlaying: boolean = false;
  currentVibe: string = 'minimalist ambient drone, quiet';
  targetVibe: string = 'minimalist ambient drone, quiet';
  vibeBlend: number = 1.0;
  nextNoteTime: number = 0;
  timerID: number | null = null;
  intensity: number = 1.0;
  beatSync: boolean = false;
  beatCount: number = 0;
  crossfadeSpeed: number = 0.02;
  forceScale: string | null = null;
  globalBpm: number | null = null;
  currentBpm: number = 60.0;
  masterCompressor: DynamicsCompressorNode;
  analyser: AnalyserNode;
  useAutoGain: boolean = false;
  mediaRecorder: MediaRecorder | null = null;
  audioChunks: Blob[] = [];
  streamDestination: MediaStreamAudioDestinationNode;
  
  // FX and Spatial Audio
  delayAmount: number = 0.4;
  delayTime: number = 0.33;
  spatialPan: number = 0; // -1 to 1 (left to right)
  
  // Scales (intervals from root)
  scales: Record<string, number[]> = {
    major: [0, 2, 4, 5, 7, 9, 11],
    minor: [0, 2, 3, 5, 7, 8, 10],
    pentatonic: [0, 2, 4, 7, 9],
    cyberpunk: [0, 3, 7, 8, 10], // Phrygian dominant-ish
    drone: [0, 7], // Just roots and fifths
    melancholic: [0, 2, 3, 7, 8], // Minor pentatonic-ish
    dissonant: [0, 1, 6, 7, 11], // For fear/disgust
    tribal: [0, 3, 5, 7, 10] // Minor pentatonic
  };

  constructor() {
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    this.masterCompressor = this.audioContext.createDynamicsCompressor();
    this.masterCompressor.threshold.value = -24;
    this.masterCompressor.knee.value = 30;
    this.masterCompressor.ratio.value = 12;
    this.masterCompressor.attack.value = 0.003;
    this.masterCompressor.release.value = 0.25;
    
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 256;
    
    this.masterCompressor.connect(this.analyser);
    this.analyser.connect(this.audioContext.destination);
    
    this.streamDestination = this.audioContext.createMediaStreamDestination();
    this.masterCompressor.connect(this.streamDestination);
  }

  setAutoGain(enabled: boolean) {
    this.useAutoGain = enabled;
  }

  setGlobalBpm(bpm: number | null) {
    this.globalBpm = bpm;
  }

  setVibe(vibe: string) {
    if (this.targetVibe !== vibe) {
      this.currentVibe = this.targetVibe;
      this.targetVibe = vibe;
      this.vibeBlend = 0.0;
    }
  }

  setForceScale(scale: string | null) {
    this.forceScale = scale;
  }

  startRecording() {
    this.audioChunks = [];
    try {
      this.mediaRecorder = new MediaRecorder(this.streamDestination.stream);
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };
      this.mediaRecorder.start();
    } catch (e) {
      console.error("Failed to start MediaRecorder:", e);
    }
  }

  stopRecording(): Promise<string | null> {
    return new Promise((resolve) => {
      if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') {
        resolve(null);
        return;
      }

      this.mediaRecorder.onstop = () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        const url = URL.createObjectURL(audioBlob);
        resolve(url);
      };
      
      this.mediaRecorder.stop();
    });
  }

  start() {
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
    this.isPlaying = true;
    this.nextNoteTime = this.audioContext.currentTime + 0.1;
    
    const initialTempo = this.globalBpm !== null && this.globalBpm !== 0
      ? this.globalBpm
      : this.getTempoForVibe(this.currentVibe) * this.intensity;
    this.currentBpm = initialTempo || 60.0;

    this.scheduleNext();
  }

  stop() {
    this.isPlaying = false;
    if (this.timerID !== null) {
      clearTimeout(this.timerID);
      this.timerID = null;
    }
    if (this.audioContext.state !== 'closed') {
      this.audioContext.close();
    }
  }

  setSpatialPan(pan: number) {
    this.spatialPan = Math.max(-1, Math.min(1, pan));
  }

  setDelayAmount(amount: number) {
    this.delayAmount = Math.max(0, Math.min(1, amount));
  }

  setDelayTime(time: number) {
    this.delayTime = Math.max(0.1, Math.min(2.0, time));
  }

  playDrum(type: 'kick' | 'snare' | 'hihat' | 'tom', vol: number, time: number) {
    if (this.audioContext.state === 'closed') return;
    
    const masterGain = this.audioContext.createGain();
    const panner = this.audioContext.createStereoPanner();
    panner.pan.value = this.spatialPan;
    
    if (this.useAutoGain) {
      masterGain.connect(this.masterCompressor);
    } else {
      masterGain.connect(this.analyser);
    }
    panner.connect(masterGain);

    const osc = this.audioContext.createOscillator();
    const gain = this.audioContext.createGain();
    osc.connect(gain);
    gain.connect(panner);

    if (type === 'kick') {
      osc.frequency.setValueAtTime(150, time);
      osc.frequency.exponentialRampToValueAtTime(0.01, time + 0.5);
      gain.gain.setValueAtTime(vol, time);
      gain.gain.exponentialRampToValueAtTime(0.01, time + 0.5);
      osc.start(time);
      osc.stop(time + 0.5);
    } else if (type === 'snare') {
      // Noise buffer for snare
      const bufferSize = this.audioContext.sampleRate * 0.2;
      const buffer = this.audioContext.createBuffer(1, bufferSize, this.audioContext.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = Math.random() * 2 - 1;
      }
      const noise = this.audioContext.createBufferSource();
      noise.buffer = buffer;
      const noiseFilter = this.audioContext.createBiquadFilter();
      noiseFilter.type = 'highpass';
      noiseFilter.frequency.value = 1000;
      noise.connect(noiseFilter);
      noiseFilter.connect(gain);
      
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(250, time);
      gain.gain.setValueAtTime(vol, time);
      gain.gain.exponentialRampToValueAtTime(0.01, time + 0.2);
      
      osc.start(time);
      noise.start(time);
      osc.stop(time + 0.2);
      noise.stop(time + 0.2);
    } else if (type === 'hihat') {
      const bufferSize = this.audioContext.sampleRate * 0.1;
      const buffer = this.audioContext.createBuffer(1, bufferSize, this.audioContext.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = Math.random() * 2 - 1;
      }
      const noise = this.audioContext.createBufferSource();
      noise.buffer = buffer;
      const noiseFilter = this.audioContext.createBiquadFilter();
      noiseFilter.type = 'bandpass';
      noiseFilter.frequency.value = 10000;
      noise.connect(noiseFilter);
      noiseFilter.connect(gain);
      
      gain.gain.setValueAtTime(vol * 0.5, time);
      gain.gain.exponentialRampToValueAtTime(0.01, time + 0.05);
      
      noise.start(time);
      noise.stop(time + 0.05);
    }
  }

  playNote(freq: number, type: OscillatorType, duration: number, vol: number, attack: number, time: number) {
    if (this.audioContext.state === 'closed') return;
    
    // Create multiple oscillators for a thicker soundscape
    const numOscs = 4;
    const masterGain = this.audioContext.createGain();
    const panner = this.audioContext.createStereoPanner();
    panner.pan.value = this.spatialPan;
    
    if (this.useAutoGain) {
      masterGain.connect(this.masterCompressor);
    } else {
      masterGain.connect(this.analyser);
    }
    
    panner.connect(masterGain);
    
    const now = time;
    masterGain.gain.setValueAtTime(0, now);
    masterGain.gain.linearRampToValueAtTime(vol, now + attack);
    masterGain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    // Add a subtle reverb effect using a convolver or just delay
    const delay = this.audioContext.createDelay();
    delay.delayTime.value = this.delayTime;
    const feedback = this.audioContext.createGain();
    feedback.gain.value = this.delayAmount;
    delay.connect(feedback);
    feedback.connect(delay);
    delay.connect(panner);

    for (let i = 0; i < numOscs; i++) {
      const osc = this.audioContext.createOscillator();
      const filter = this.audioContext.createBiquadFilter();
      
      osc.type = i % 2 === 0 ? type : 'sine';
      osc.frequency.value = freq * (1 + (i * 0.008)); // Slight detune
      
      filter.type = 'lowpass';
      filter.frequency.value = freq * 2;
      filter.frequency.linearRampToValueAtTime(freq * 6, now + attack);
      filter.frequency.linearRampToValueAtTime(freq * 1.5, now + duration);
      
      osc.connect(filter);
      filter.connect(panner);
      filter.connect(delay); // Send to delay for space
      
      osc.start(now);
      osc.stop(now + duration);
    }
  }

  getTempoForVibe(vibe: string): number {
    if (vibe.includes('tribal') || vibe.includes('rhythmic')) return 100;
    if (vibe.includes('cyberpunk') || vibe.includes('electronic')) return 60;
    return 40;
  }

  scheduleNext() {
    if (!this.isPlaying) return;
    
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
    
    while (this.nextNoteTime < this.audioContext.currentTime + 0.5) {
      if (this.vibeBlend < 1.0) {
        this.vibeBlend += this.crossfadeSpeed; // crossfade over notes for a smoother transition
        if (this.vibeBlend > 1.0) this.vibeBlend = 1.0;
      }

      if (this.vibeBlend < 1.0) {
        // Equal power crossfade for smoother audio blending
        const currentWeight = Math.cos(this.vibeBlend * 0.5 * Math.PI);
        const targetWeight = Math.sin(this.vibeBlend * 0.5 * Math.PI);
        this.generateTickForVibe(this.currentVibe, currentWeight, this.nextNoteTime, this.beatCount);
        this.generateTickForVibe(this.targetVibe, targetWeight, this.nextNoteTime, this.beatCount);
      } else {
        this.generateTickForVibe(this.targetVibe, 1.0, this.nextNoteTime, this.beatCount);
      }
      
      this.beatCount = (this.beatCount + 1) % 4;
      
      // Smoothly interpolate tempo
      const currentTempo = this.getTempoForVibe(this.currentVibe);
      const targetTempo = this.getTempoForVibe(this.targetVibe);
      let targetBpm = (currentTempo * (1 - this.vibeBlend) + targetTempo * this.vibeBlend) * this.intensity;
      
      if (this.globalBpm !== null && this.globalBpm !== 0) {
        targetBpm = this.globalBpm;
      }

      // Smoothly interpolate currentBpm towards targetBpm
      const lerpFactor = 0.08; // smooth transition over beats/ticks
      this.currentBpm = this.currentBpm + (targetBpm - this.currentBpm) * lerpFactor;
      
      const secondsPerBeat = 60.0 / this.currentBpm;
      this.nextNoteTime += secondsPerBeat; // Quarter notes
    }
    
    this.timerID = window.setTimeout(() => this.scheduleNext(), 50);
  }

  generateTickForVibe(vibe: string, weight: number, time: number, beatCount: number) {
    if (weight <= 0.01) return;
    
    const isCyberpunk = vibe.includes('cyberpunk') || vibe.includes('electronic');
    const isTribal = vibe.includes('tribal') || vibe.includes('rhythmic') || vibe.includes('happy');
    const isDriving = vibe.includes('driving');
    const isAcoustic = vibe.includes('acoustic') || vibe.includes('guitar');
    const isAmbient = vibe.includes('ambient') || vibe.includes('drone');
    const isSad = vibe.includes('sad') || vibe.includes('melancholy');
    const isTense = vibe.includes('angry') || vibe.includes('fear') || vibe.includes('disgust');
    
    let scale = this.scales.pentatonic;
    let baseNote = 48; // C3
    let oscType: OscillatorType = 'sine';
    let vol = 0.08;
    let duration = 6.0; // Longer durations for soundscape
    let attack = 3.0;

    if (isCyberpunk) {
      scale = this.scales.cyberpunk;
      baseNote = 36; // C2
      oscType = 'sawtooth';
      vol = 0.04;
      duration = 4.0;
      attack = 2.0;
    } else if (isTribal || isDriving) {
      scale = this.scales.tribal;
      baseNote = 43; // G2
      oscType = 'square';
      vol = 0.06;
      duration = 1.5;
      attack = 0.1;
    } else if (isSad) {
      scale = this.scales.melancholic;
      baseNote = 48;
      oscType = 'sine';
      vol = 0.08;
      duration = 8.0;
      attack = 4.0;
    } else if (isTense) {
      scale = this.scales.dissonant;
      baseNote = 36;
      oscType = 'sawtooth';
      vol = 0.05;
      duration = 5.0;
      attack = 1.5;
    } else if (isAcoustic) {
      scale = this.scales.major;
      baseNote = 48;
      oscType = 'sine';
      vol = 0.08;
      duration = 5.0;
      attack = 2.0;
    } else if (isAmbient) {
      scale = this.scales.drone;
      baseNote = 36;
      oscType = 'sine';
      vol = 0.12;
      duration = 10.0;
      attack = 5.0;
    }

    if (this.forceScale && this.scales[this.forceScale]) {
      scale = this.scales[this.forceScale];
    }

    vol *= weight; // Apply crossfade weight
    
    let noteProb = Math.min(1.0, 0.8 * this.intensity);
    let droneProb = Math.min(1.0, 0.5 * this.intensity);
    const forceBeat = this.beatSync && (isTribal || isDriving);
    
    let drumProb = 0;
    
    if (forceBeat) {
      noteProb = 1.0;
      droneProb = beatCount === 0 ? 1.0 : 0.0;
      drumProb = 1.0;
      if (beatCount !== 0) {
        duration = 0.5;
        attack = 0.05;
        vol *= 0.5;
      } else {
        duration = 1.5;
        attack = 0.05;
        vol *= 1.2;
      }
    } else if (this.beatSync && isCyberpunk) {
      drumProb = 1.0;
    }

    if (drumProb > 0) {
      const drumVol = 0.15 * weight * this.intensity;
      if (isCyberpunk) {
        if (beatCount === 0 || beatCount === 2) this.playDrum('kick', drumVol * 1.5, time);
        if (beatCount === 1 || beatCount === 3) {
           this.playDrum('snare', drumVol, time);
           this.playDrum('hihat', drumVol * 0.5, time + 0.25);
        }
        if (beatCount === 0) this.playDrum('hihat', drumVol * 0.5, time + 0.25);
      } else if (isDriving) {
        if (beatCount === 0 || beatCount === 2) this.playDrum('kick', drumVol * 1.2, time);
        if (beatCount === 1 || beatCount === 3) this.playDrum('snare', drumVol, time);
        this.playDrum('hihat', drumVol * 0.4, time);
        this.playDrum('hihat', drumVol * 0.3, time + 0.25);
      } else if (isTribal) {
        if (beatCount === 0) this.playDrum('kick', drumVol * 1.2, time);
        if (beatCount === 2) this.playDrum('kick', drumVol * 0.8, time + 0.1);
        this.playDrum('snare', drumVol * 0.6, time + 0.3);
      }
    }

    // Randomly play a note from the scale
    if (Math.random() < noteProb) {
      const noteIndex = scale[Math.floor(Math.random() * scale.length)];
      const freq = 440 * Math.pow(2, (baseNote + noteIndex - 69) / 12);
      this.playNote(freq, oscType, duration, vol, attack, time);
    }
    
    // Add a bass drone
    if (Math.random() < droneProb) {
      const bassFreq = 440 * Math.pow(2, (baseNote - 12 - 69) / 12);
      this.playNote(bassFreq, 'sine', duration * 2, vol * 1.5, attack * 2, time);
    }
  }
}
