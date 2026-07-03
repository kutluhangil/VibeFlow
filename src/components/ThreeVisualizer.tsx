import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { ProceduralMusicEngine } from '../engine/ProceduralMusicEngine';

interface ThreeVisualizerProps {
  engine: ProceduralMusicEngine | null;
  mode: 'default' | 'wireframe' | 'particle';
  sensitivity: number;
  themeColor: string;
}

export const ThreeVisualizer: React.FC<ThreeVisualizerProps> = ({ engine, mode, sensitivity, themeColor }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const particlesRef = useRef<THREE.Points | null>(null);
  const reqRef = useRef<number>(0);
  const dataArrayRef = useRef<Uint8Array | null>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    // Setup scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(75, mountRef.current.clientWidth / mountRef.current.clientHeight, 0.1, 1000);
    camera.position.z = 50;
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Create particle system
    const particleCount = 1000;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const originalPositions = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount * 3; i += 3) {
      // Sphere distribution
      const r = 20 + Math.random() * 20;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos((Math.random() * 2) - 1);

      positions[i] = r * Math.sin(phi) * Math.cos(theta);
      positions[i + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i + 2] = r * Math.cos(phi);
      
      originalPositions[i] = positions[i];
      originalPositions[i+1] = positions[i+1];
      originalPositions[i+2] = positions[i+2];
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('originalPosition', new THREE.BufferAttribute(originalPositions, 3));

    const material = new THREE.PointsMaterial({
      color: new THREE.Color(themeColor),
      size: 0.5,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending
    });

    const particles = new THREE.Points(geometry, material);
    scene.add(particles);
    particlesRef.current = particles;

    const handleResize = () => {
      if (!mountRef.current || !camera || !renderer) return;
      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (reqRef.current) cancelAnimationFrame(reqRef.current);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      geometry.dispose();
      material.dispose();
      renderer.dispose();
    };
  }, []); // Only run once on mount

  useEffect(() => {
    if (particlesRef.current) {
      (particlesRef.current.material as THREE.PointsMaterial).color = new THREE.Color(themeColor);
    }
  }, [themeColor]);

  useEffect(() => {
    if (engine && !dataArrayRef.current) {
      dataArrayRef.current = new Uint8Array(engine.analyser.frequencyBinCount);
    }

    const animate = () => {
      reqRef.current = requestAnimationFrame(animate);

      if (!sceneRef.current || !cameraRef.current || !rendererRef.current || !particlesRef.current) return;

      const particles = particlesRef.current;
      const positions = particles.geometry.attributes.position.array as Float32Array;
      const originalPositions = particles.geometry.attributes.originalPosition.array as Float32Array;

      // Update rotation
      particles.rotation.y += 0.002;
      particles.rotation.x += 0.001;

      // Audio reactive
      let audioForce = 0;
      if (engine && dataArrayRef.current) {
        engine.analyser.getByteFrequencyData(dataArrayRef.current);
        const sum = dataArrayRef.current.reduce((a, b) => a + b, 0);
        const avg = sum / dataArrayRef.current.length;
        audioForce = (avg / 255.0) * sensitivity; // 0 to 1 * sensitivity
      }

      // Morph particles based on audio
      const time = performance.now() * 0.001;
      
      for (let i = 0; i < positions.length; i += 3) {
        const ox = originalPositions[i];
        const oy = originalPositions[i + 1];
        const oz = originalPositions[i + 2];
        
        // Calculate distance from center
        const dist = Math.sqrt(ox*ox + oy*oy + oz*oz);
        const normalX = ox / dist;
        const normalY = oy / dist;
        const normalZ = oz / dist;

        let force = audioForce * 15;
        
        if (mode === 'wireframe') {
            force = audioForce * 5;
            positions[i] = ox + Math.sin(time * 2 + i) * force;
            positions[i+1] = oy + Math.cos(time * 2 + i) * force;
            positions[i+2] = oz + Math.sin(time * 2 + i) * force;
        } else if (mode === 'particle') {
            // Explosive effect
            positions[i] = ox + normalX * force * (Math.random() * 2);
            positions[i+1] = oy + normalY * force * (Math.random() * 2);
            positions[i+2] = oz + normalZ * force * (Math.random() * 2);
        } else {
            // Default gentle pulse
            positions[i] = ox + normalX * force;
            positions[i+1] = oy + normalY * force;
            positions[i+2] = oz + normalZ * force;
        }
      }
      
      particles.geometry.attributes.position.needsUpdate = true;

      // Update material size based on mode and audio
      const material = particles.material as THREE.PointsMaterial;
      if (mode === 'wireframe') {
          material.size = 0.2;
          material.opacity = 0.4;
      } else {
          material.size = 0.5 + audioForce * 2;
          material.opacity = 0.6 + audioForce * 0.4;
      }

      rendererRef.current.render(sceneRef.current, cameraRef.current);
    };

    reqRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(reqRef.current);
    };
  }, [engine, mode, sensitivity]);

  return <div ref={mountRef} className="absolute inset-0 pointer-events-none" />;
};
