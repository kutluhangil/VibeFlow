export interface Snapshot {
  id: string;
  timestamp: number;
  imageUri: string;
  audioProfile: string;
  intensity?: number;
  beatSync?: boolean;
  autoGain?: boolean;
  crossfadeSpeed?: number;
  visualizerSensitivity?: number;
  forceScale?: string | null;
  globalBpm?: number | null;
  isAutoCaptured?: boolean;
}
