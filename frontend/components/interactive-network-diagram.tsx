"use client";

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Play, Pause, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';

interface Neuron {
  id: string;
  x: number;
  y: number;
  value: number;
  active: boolean;
  type: 'input' | 'hidden' | 'output';
  layer: number;
}

interface Connection {
  from: string;
  to: string;
  weight: number;
  active: boolean;
}

interface LayerInfo {
  name: string;
  type: 'input' | 'encoder' | 'decoder' | 'output';
  neurons: number;
  description: string;
  operation: string;
}

interface InteractiveNetworkDiagramProps {
  currentLayer: number;
  onLayerChange: (layer: number) => void;
  inputImage?: string;
  outputImage?: string;
  modelType?: string;
}

const InteractiveNetworkDiagram: React.FC<InteractiveNetworkDiagramProps> = ({
  currentLayer,
  onLayerChange,
  inputImage,
  outputImage,
  modelType = 'segformer'
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [neurons, setNeurons] = useState<Neuron[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [isAnimating, setIsAnimating] = useState(false);
  const [hoveredNeuron, setHoveredNeuron] = useState<string | null>(null);
  const [selectedNeuron, setSelectedNeuron] = useState<string | null>(null);
  const animationRef = useRef<number>();

  // Simplified layer definitions for SegFormer
  const layers: LayerInfo[] = [
    { name: "Input", type: "input", neurons: 8, description: "RGB Image", operation: "Preprocessing" },
    { name: "Stage 1", type: "encoder", neurons: 12, description: "MiT Block", operation: "Self-Attention" },
    { name: "Stage 2", type: "encoder", neurons: 16, description: "MiT Block", operation: "Multi-Scale" },
    { name: "Stage 3", type: "encoder", neurons: 20, description: "MiT Block", operation: "Semantic" },
    { name: "Stage 4", type: "encoder", neurons: 24, description: "MiT Block", operation: "Global Context" },
    { name: "Decoder", type: "decoder", neurons: 16, description: "MLP Fusion", operation: "Feature Fusion" },
    { name: "Output", type: "output", neurons: 10, description: "Classes", operation: "Classification" }
  ];

  // Initialize network structure
  useEffect(() => {
    initializeNetwork();
  }, []);

  // Animation loop
  useEffect(() => {
    if (isAnimating) {
      startAnimation();
    } else {
      stopAnimation();
    }
    return () => stopAnimation();
  }, [isAnimating, currentLayer]);

  // Canvas drawing
  useEffect(() => {
    drawNetwork();
  }, [neurons, connections, hoveredNeuron, selectedNeuron, currentLayer]);

  const initializeNetwork = () => {
    const newNeurons: Neuron[] = [];
    const newConnections: Connection[] = [];
    
    const canvasWidth = 1200;
    const canvasHeight = 600;
    const layerSpacing = canvasWidth / (layers.length + 1);
    
    // Create neurons for each layer
    layers.forEach((layer, layerIndex) => {
      const x = layerSpacing * (layerIndex + 1);
      const neuronSpacing = canvasHeight / (layer.neurons + 1);
      
      for (let i = 0; i < layer.neurons; i++) {
        const y = neuronSpacing * (i + 1);
        newNeurons.push({
          id: `${layerIndex}-${i}`,
          x,
          y,
          value: Math.random(),
          active: false,
          type: layer.type === 'input' ? 'input' : layer.type === 'output' ? 'output' : 'hidden',
          layer: layerIndex
        });
      }
    });

    // Create connections between adjacent layers
    for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex++) {
      const currentLayerNeurons = newNeurons.filter(n => n.layer === layerIndex);
      const nextLayerNeurons = newNeurons.filter(n => n.layer === layerIndex + 1);
      
      currentLayerNeurons.forEach(fromNeuron => {
        nextLayerNeurons.forEach(toNeuron => {
          // Create connections with some sparsity for attention mechanism
          const shouldConnect = layerIndex < 2 ? true : Math.random() > 0.3;
          if (shouldConnect) {
            newConnections.push({
              from: fromNeuron.id,
              to: toNeuron.id,
              weight: (Math.random() - 0.5) * 2,
              active: false
            });
          }
        });
      });
    }

    setNeurons(newNeurons);
    setConnections(newConnections);
  };

  const startAnimation = () => {
    const animate = () => {
      // Activate neurons layer by layer
      setNeurons(prev => prev.map(neuron => ({
        ...neuron,
        active: neuron.layer <= currentLayer,
        value: neuron.layer <= currentLayer ? Math.random() * 0.8 + 0.2 : 0
      })));

      // Activate connections
      setConnections(prev => prev.map(conn => {
        const fromNeuron = neurons.find(n => n.id === conn.from);
        const toNeuron = neurons.find(n => n.id === conn.to);
        return {
          ...conn,
          active: fromNeuron?.layer === currentLayer && toNeuron?.layer === currentLayer + 1
        };
      }));

      animationRef.current = requestAnimationFrame(animate);
    };
    animate();
  };

  const stopAnimation = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  };

  const drawNetwork = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw connections first
    connections.forEach(conn => {
      const fromNeuron = neurons.find(n => n.id === conn.from);
      const toNeuron = neurons.find(n => n.id === conn.to);
      
      if (fromNeuron && toNeuron) {
        ctx.beginPath();
        ctx.moveTo(fromNeuron.x, fromNeuron.y);
        ctx.lineTo(toNeuron.x, toNeuron.y);
        
        // Connection styling based on weight and activity
        const opacity = conn.active ? 0.4 : 0.1; // Reduced opacity for connections
        const weight = Math.abs(conn.weight);
        const color = conn.weight > 0 ? `rgba(59, 130, 246, ${opacity})` : `rgba(239, 68, 68, ${opacity})`;
        
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(0.3, weight * 2); // Reduced line width
        ctx.stroke();

        // Draw weight value on hover
        if (hoveredNeuron === conn.from || hoveredNeuron === conn.to) {
          const midX = (fromNeuron.x + toNeuron.x) / 2;
          const midY = (fromNeuron.y + toNeuron.y) / 2;
          
          ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
          ctx.fillRect(midX - 15, midY - 8, 30, 16);
          ctx.fillStyle = 'white';
          ctx.font = '10px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(conn.weight.toFixed(2), midX, midY + 3);
        }
      }
    });

    // Draw neurons
    neurons.forEach(neuron => {
      const isHovered = hoveredNeuron === neuron.id;
      const isSelected = selectedNeuron === neuron.id;
      const isCurrentLayer = neuron.layer === currentLayer;
      
      // Neuron circle
      ctx.beginPath();
      ctx.arc(neuron.x, neuron.y, isHovered ? 12 : 8, 0, 2 * Math.PI);
      
      // Color based on type and activity
      let fillColor = 'rgba(156, 163, 175, 0.3)'; // Default gray - reduced opacity
      if (neuron.type === 'input') fillColor = 'rgba(59, 130, 246, 0.5)'; // Blue - reduced opacity
      else if (neuron.type === 'output') fillColor = 'rgba(168, 85, 247, 0.5)'; // Purple - reduced opacity
      else if (neuron.active) fillColor = 'rgba(34, 197, 94, 0.5)'; // Green - reduced opacity
      
      if (isCurrentLayer) {
        fillColor = 'rgba(251, 191, 36, 0.6)'; // Yellow for current layer - reduced opacity
      }
      
      ctx.fillStyle = fillColor;
      ctx.fill();
      
      // Border
      ctx.strokeStyle = isSelected ? 'rgba(239, 68, 68, 1)' : 'rgba(0, 0, 0, 0.3)';
      ctx.lineWidth = isSelected ? 3 : 1;
      ctx.stroke();

      // Value indicator (brightness)
      if (neuron.active && neuron.value > 0) {
        ctx.beginPath();
        ctx.arc(neuron.x, neuron.y, 3, 0, 2 * Math.PI); // Smaller indicator
        ctx.fillStyle = `rgba(255, 255, 255, ${neuron.value * 0.6})`; // Reduced brightness
        ctx.fill();
      }

      // Neuron ID on hover
      if (isHovered || isSelected) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(neuron.x - 20, neuron.y - 25, 40, 15);
        ctx.fillStyle = 'white';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(neuron.id, neuron.x, neuron.y - 17);
        ctx.fillText(`v: ${neuron.value.toFixed(2)}`, neuron.x, neuron.y - 35);
      }
    });

    // Draw layer labels
    layers.forEach((layer, index) => {
      const layerNeurons = neurons.filter(n => n.layer === index);
      if (layerNeurons.length > 0) {
        const x = layerNeurons[0].x;
        const isCurrentLayer = index === currentLayer;
        
        // Layer background
        ctx.fillStyle = isCurrentLayer ? 'rgba(251, 191, 36, 0.1)' : 'rgba(0, 0, 0, 0.05)'; // Much more subtle background
        ctx.fillRect(x - 40, 10, 80, 60);
        
        // Layer text
        ctx.fillStyle = isCurrentLayer ? 'rgba(251, 191, 36, 0.9)' : 'rgba(0, 0, 0, 0.6)'; // Reduced text opacity
        ctx.font = isCurrentLayer ? 'bold 12px sans-serif' : '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(layer.name, x, 25);
        ctx.font = '8px sans-serif';
        ctx.fillText(`${layer.neurons} neurons`, x, 40);
        ctx.fillText(layer.operation, x, 55);
      }
    });

    // Draw input/output images
    if (inputImage && neurons.length > 0) {
      const inputNeurons = neurons.filter(n => n.type === 'input');
      if (inputNeurons.length > 0) {
        const img = new Image();
        img.onload = () => {
          ctx.drawImage(img, inputNeurons[0].x - 60, inputNeurons[0].y - 30, 60, 60);
        };
        img.src = inputImage;
      }
    }

    if (outputImage && neurons.length > 0) {
      const outputNeurons = neurons.filter(n => n.type === 'output');
      if (outputNeurons.length > 0) {
        const img = new Image();
        img.onload = () => {
          ctx.drawImage(img, outputNeurons[0].x + 20, outputNeurons[0].y - 30, 60, 60);
        };
        img.src = outputImage;
      }
    }
  };

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Find clicked neuron
    const clickedNeuron = neurons.find(neuron => {
      const distance = Math.sqrt((neuron.x - x) ** 2 + (neuron.y - y) ** 2);
      return distance <= 12;
    });

    if (clickedNeuron) {
      setSelectedNeuron(clickedNeuron.id);
      onLayerChange(clickedNeuron.layer);
    } else {
      setSelectedNeuron(null);
    }
  };

  const handleCanvasMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Find hovered neuron
    const hoveredNeuron = neurons.find(neuron => {
      const distance = Math.sqrt((neuron.x - x) ** 2 + (neuron.y - y) ** 2);
      return distance <= 12;
    });

    setHoveredNeuron(hoveredNeuron?.id || null);
    canvas.style.cursor = hoveredNeuron ? 'pointer' : 'default';
  };

  const handlePlay = () => setIsAnimating(!isAnimating);
  const handleReset = () => {
    setIsAnimating(false);
    onLayerChange(0);
    setSelectedNeuron(null);
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Interactive Network Diagram</span>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{layers[currentLayer]?.name}</Badge>
            <Badge variant="secondary">{modelType.toUpperCase()}</Badge>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Controls */}
        <div className="flex flex-wrap items-center gap-4 mb-6">
          <Button onClick={handlePlay} variant="outline" size="sm">
            {isAnimating ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
            {isAnimating ? 'Pause' : 'Play'}
          </Button>
          <Button onClick={handleReset} variant="outline" size="sm">
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </Button>
          <div className="flex items-center gap-2">
            <span className="text-sm">Layer:</span>
            <span className="font-mono text-sm">{currentLayer + 1}/{layers.length}</span>
          </div>
        </div>

        {/* Canvas */}
        <div className="border rounded-lg overflow-hidden bg-gradient-to-br from-gray-50 to-gray-100">
          <canvas
            ref={canvasRef}
            width={1200}
            height={600}
            className="w-full h-auto cursor-crosshair"
            onClick={handleCanvasClick}
            onMouseMove={handleCanvasMouseMove}
          />
        </div>

        {/* Layer Information */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-muted/50 rounded-lg">
            <h4 className="font-semibold mb-2">Current Layer</h4>
            <p className="text-sm text-muted-foreground mb-1">
              <strong>{layers[currentLayer]?.name}</strong>
            </p>
            <p className="text-xs text-muted-foreground">
              {layers[currentLayer]?.description}
            </p>
          </div>
          
          <div className="p-4 bg-muted/50 rounded-lg">
            <h4 className="font-semibold mb-2">Interaction</h4>
            <p className="text-xs text-muted-foreground mb-1">
              • Click neurons to select layers
            </p>
            <p className="text-xs text-muted-foreground mb-1">
              • Hover to see connections
            </p>
            <p className="text-xs text-muted-foreground">
              • Play to see data flow
            </p>
          </div>
        </div>

        {/* Legend */}
        <div className="mt-4 flex flex-wrap gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-blue-400 opacity-60"></div>
            <span>Input Layer</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-green-400 opacity-60"></div>
            <span>Hidden Layers</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-purple-400 opacity-60"></div>
            <span>Output Layer</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-yellow-400 opacity-70"></div>
            <span>Current Layer</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-1 bg-blue-400 opacity-50"></div>
            <span>Positive Weight</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-1 bg-red-400 opacity-50"></div>
            <span>Negative Weight</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default InteractiveNetworkDiagram;