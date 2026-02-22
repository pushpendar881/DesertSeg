"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Play, Pause, RotateCcw, Layers, Network } from 'lucide-react';
import dynamic from 'next/dynamic';

// Dynamically import network diagram
const InteractiveNetworkDiagram = dynamic(() => import('./interactive-network-diagram'), {
  ssr: false,
  loading: () => <div className="flex items-center justify-center p-8"><div className="animate-spin w-6 h-6 border-2 border-primary border-t-transparent rounded-full"></div></div>
});

interface LayerVisualization {
  name: string;
  type: 'encoder' | 'decoder' | 'input' | 'output';
  stage: number;
  channels: number;
  resolution: string;
  description: string;
  features: string[];
}

interface SegFormerExplainerProps {
  inputImage?: string;
  segmentationResult?: any;
  modelType?: 'segformer' | 'segformer-b1' | 'segformer-b10';
}

const SegFormerExplainer: React.FC<SegFormerExplainerProps> = ({
  inputImage,
  segmentationResult,
  modelType = 'segformer'
}) => {
  const [currentLayer, setCurrentLayer] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState([1000]);
  const [viewMode, setViewMode] = useState<'flow' | 'network'>('flow');
  const [featureVisualizationCache, setFeatureVisualizationCache] = useState<Record<number, string>>({});
  const [isMounted, setIsMounted] = useState(false);

  // Simplified SegFormer architecture layers
  const layers: LayerVisualization[] = [
    {
      name: "Input Image",
      type: "input",
      stage: 0,
      channels: 3,
      resolution: "H×W",
      description: "RGB input image with desert terrain features",
      features: ["Raw pixels", "Color channels", "Spatial info"]
    },
    {
      name: "Patch Embedding",
      type: "encoder",
      stage: 1,
      channels: 64,
      resolution: "H/4×W/4",
      description: "Converts image patches into token embeddings",
      features: ["4×4 patches", "Linear projection", "Overlapped merging"]
    },
    {
      name: "Stage 1 - MiT Block",
      type: "encoder",
      stage: 1,
      channels: 64,
      resolution: "H/4×W/4",
      description: "Hierarchical attention with efficient self-attention",
      features: ["Self-attention", "Spatial reduction", "Feed-forward"]
    },
    {
      name: "Stage 2 - MiT Block",
      type: "encoder",
      stage: 2,
      channels: 128,
      resolution: "H/8×W/8",
      description: "Multi-scale feature learning with spatial reduction",
      features: ["Multi-scale attention", "Feature pyramid", "Context modeling"]
    },
    {
      name: "Stage 3 - MiT Block",
      type: "encoder",
      stage: 3,
      channels: 320,
      resolution: "H/16×W/16",
      description: "High-level semantic feature extraction",
      features: ["Global context", "Semantic features", "Abstract patterns"]
    },
    {
      name: "Stage 4 - MiT Block",
      type: "encoder",
      stage: 4,
      channels: 512,
      resolution: "H/32×W/32",
      description: "Deepest features with maximum receptive field",
      features: ["Global relationships", "Abstract features", "Full attention"]
    },
    {
      name: "MLP Decoder",
      type: "decoder",
      stage: 5,
      channels: 256,
      resolution: "H/4×W/4",
      description: "Lightweight decoder aggregates multi-scale features",
      features: ["Feature fusion", "Channel alignment", "Upsampling"]
    },
    {
      name: "Classification Head",
      type: "decoder",
      stage: 6,
      channels: 10,
      resolution: "H×W",
      description: "Final pixel-wise classification for terrain classes",
      features: ["Per-pixel classification", "Class probabilities", "Softmax"]
    },
    {
      name: "Output Segmentation",
      type: "output",
      stage: 7,
      channels: 10,
      resolution: "H×W",
      description: "Final segmentation mask with terrain predictions",
      features: ["Trees", "Bushes", "Grass", "Rocks", "Sky"]
    }
  ];

  // Auto-play functionality
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentLayer((prev) => (prev + 1) % layers.length);
      }, playSpeed[0]);
    }
    return () => clearInterval(interval);
  }, [isPlaying, playSpeed, layers.length]);

  // Mount effect for client-side only features
  useEffect(() => {
    setIsMounted(true);
  }, []);

  const handlePlay = () => setIsPlaying(!isPlaying);
  const handleReset = () => {
    setCurrentLayer(0);
    setIsPlaying(false);
  };

  const getLayerColor = (type: string) => {
    switch (type) {
      case 'input': return 'bg-blue-50 border-blue-400 shadow-blue-100';
      case 'encoder': return 'bg-green-50 border-green-400 shadow-green-100';
      case 'decoder': return 'bg-orange-50 border-orange-400 shadow-orange-100';
      case 'output': return 'bg-purple-50 border-purple-400 shadow-purple-100';
      default: return 'bg-gray-50 border-gray-400 shadow-gray-100';
    }
  };

  const getStageColor = (stage: number) => {
    const colors = [
      'bg-blue-500', 'bg-green-500', 'bg-yellow-500', 
      'bg-orange-500', 'bg-red-500', 'bg-purple-500',
      'bg-pink-500', 'bg-indigo-500'
    ];
    return colors[stage % colors.length];
  };

  const generateFeatureVisualization = (layer: LayerVisualization, layerIndex: number) => {
    // Return cached version if available
    if (featureVisualizationCache[layerIndex]) {
      return featureVisualizationCache[layerIndex];
    }

    // Only generate on client side
    if (typeof window === 'undefined' || !isMounted) {
      return '';
    }

    try {
      // Generate a procedural feature map visualization based on layer properties
      const canvas = document.createElement('canvas');
      canvas.width = 200;
      canvas.height = 200;
      const ctx = canvas.getContext('2d');
      
      if (!ctx) return '';
      
      // Clear canvas
      ctx.fillStyle = '#f8f9fa';
      ctx.fillRect(0, 0, 200, 200);
      
      // Generate different patterns based on layer type and stage
      const gridSize = Math.max(4, Math.min(32, 200 / Math.sqrt(layer.channels)));
      const cols = Math.floor(200 / gridSize);
      const rows = Math.floor(200 / gridSize);
      
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const x = j * gridSize;
          const y = i * gridSize;
          
          // Generate color based on layer properties
          let intensity = 0;
          if (layer.type === 'input') {
            intensity = Math.random() * 255;
          } else if (layer.type === 'encoder') {
            // Simulate attention patterns
            const centerX = cols / 2;
            const centerY = rows / 2;
            const distance = Math.sqrt((j - centerX) ** 2 + (i - centerY) ** 2);
            intensity = Math.max(0, 255 - (distance * 20)) * (0.5 + Math.random() * 0.5);
          } else if (layer.type === 'decoder') {
            // Simulate upsampling patterns
            intensity = (Math.sin(i * 0.5) + Math.cos(j * 0.5)) * 127 + 128;
          } else {
            // Output patterns
            intensity = Math.random() > 0.7 ? 255 : 50;
          }
          
          const hue = (layer.stage * 60) % 360;
          ctx.fillStyle = `hsla(${hue}, 70%, ${50 + intensity / 10}%, 0.8)`;
          ctx.fillRect(x, y, gridSize - 1, gridSize - 1);
        }
      }
      
      const dataUrl = canvas.toDataURL();
      
      // Cache the result
      setFeatureVisualizationCache(prev => ({
        ...prev,
        [layerIndex]: dataUrl
      }));
      
      return dataUrl;
    } catch (error) {
      console.error('Error generating feature visualization:', error);
      return '';
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Network className="w-6 h-6" />
            SegFormer Architecture Explainer
            <Badge variant="outline" className="ml-2">
              {modelType.toUpperCase()}
            </Badge>
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Interactive visualization of SegFormer's hierarchical transformer architecture for semantic segmentation
          </p>
        </CardHeader>
      </Card>

      {/* View Mode Tabs */}
      <Card>
        <CardContent className="pt-6">
          <Tabs value={viewMode} onValueChange={(value) => setViewMode(value as any)}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="flow" className="flex items-center gap-2">
                <Layers className="w-4 h-4" />
                Layer Flow
              </TabsTrigger>
              <TabsTrigger value="network" className="flex items-center gap-2">
                <Network className="w-4 h-4" />
                Network Diagram
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </CardContent>
      </Card>

      {/* Controls */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap items-center gap-4">
            <Button onClick={handlePlay} variant="outline" size="sm">
              {isPlaying ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
              {isPlaying ? 'Pause' : 'Play'}
            </Button>
            <Button onClick={handleReset} variant="outline" size="sm">
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset
            </Button>
            <div className="flex items-center gap-2">
              <span className="text-sm">Speed:</span>
              <Slider
                value={playSpeed}
                onValueChange={setPlaySpeed}
                max={2000}
                min={200}
                step={200}
                className="w-24"
              />
              <span className="text-xs text-muted-foreground">{playSpeed[0]}ms</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm">Layer:</span>
              <span className="font-mono text-sm">{currentLayer + 1}/{layers.length}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Content based on view mode */}
      {viewMode === 'flow' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Layer Flow */}
          <div className="lg:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle>Architecture Flow</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {layers.map((layer, index) => (
                    <div
                      key={index}
                      className={`p-6 rounded-xl border-2 transition-all duration-300 cursor-pointer shadow-lg ${
                        index === currentLayer 
                          ? `${getLayerColor(layer.type)} ring-4 ring-blue-300 shadow-xl transform scale-[1.02]` 
                          : 'bg-white border-gray-300 hover:bg-gray-50 hover:border-gray-400 hover:shadow-md'
                      }`}
                      onClick={() => setCurrentLayer(index)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div className={`w-10 h-10 rounded-full ${getStageColor(layer.stage)} flex items-center justify-center text-white text-sm font-bold shadow-md`}>
                            {index + 1}
                          </div>
                          <div>
                            <h3 className="font-bold text-lg text-gray-800">{layer.name}</h3>
                            <p className="text-sm text-gray-600 mt-1">{layer.description}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <Badge variant="outline" className="mb-2 font-semibold border-2">
                            {layer.channels} channels
                          </Badge>
                          <p className="text-sm font-medium text-gray-600">{layer.resolution}</p>
                        </div>
                      </div>
                      
                      {index === currentLayer && (
                        <div className="mt-6 pt-4 border-t-2 border-gray-200">
                          <h4 className="font-bold mb-3 text-gray-800">Key Features:</h4>
                          <div className="flex flex-wrap gap-2">
                            {layer.features.map((feature, fIndex) => (
                              <Badge key={fIndex} variant="secondary" className="text-sm font-medium px-3 py-1">
                                {feature}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Current Layer Details */}
          <div className="space-y-6">
            {/* Layer Information */}
            <Card className="shadow-lg border-2">
              <CardHeader className="bg-gradient-to-r from-blue-50 to-purple-50">
                <CardTitle className="text-lg">Current Layer</CardTitle>
              </CardHeader>
              <CardContent className="pt-6">
                <div className="space-y-4">
                  <div>
                    <h3 className="font-bold text-xl text-gray-800">{layers[currentLayer].name}</h3>
                    <Badge className={`mt-3 px-4 py-2 text-sm font-bold ${getLayerColor(layers[currentLayer].type)}`}>
                      {layers[currentLayer].type.toUpperCase()}
                    </Badge>
                  </div>
                  
                  <div className="space-y-3 bg-gray-50 p-4 rounded-lg">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-semibold text-gray-700">Channels:</span>
                      <span className="font-mono text-lg font-bold text-gray-800">{layers[currentLayer].channels}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-semibold text-gray-700">Resolution:</span>
                      <span className="font-mono text-lg font-bold text-gray-800">{layers[currentLayer].resolution}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-semibold text-gray-700">Stage:</span>
                      <span className="font-mono text-lg font-bold text-gray-800">{layers[currentLayer].stage}</span>
                    </div>
                  </div>

                  <div className="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-400">
                    <p className="text-sm font-medium text-gray-700 leading-relaxed">{layers[currentLayer].description}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {viewMode === 'network' && (
        <InteractiveNetworkDiagram
          currentLayer={currentLayer}
          onLayerChange={setCurrentLayer}
          inputImage={inputImage}
          outputImage={segmentationResult?.segmentationMask}
          modelType={modelType}
        />
      )}

      {/* Layer Summary */}
      <Card className="shadow-lg border-2">
        <CardHeader className="bg-gradient-to-r from-gray-50 to-gray-100">
          <CardTitle className="text-lg">SegFormer Layer Summary</CardTitle>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-sm">
            <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
              <div className="w-4 h-4 bg-blue-500 rounded-full shadow-sm"></div>
              <span className="font-semibold text-gray-700">Input Layer (1)</span>
            </div>
            <div className="flex items-center gap-3 p-3 bg-green-50 rounded-lg border border-green-200">
              <div className="w-4 h-4 bg-green-500 rounded-full shadow-sm"></div>
              <span className="font-semibold text-gray-700">Encoder Stages (5)</span>
            </div>
            <div className="flex items-center gap-3 p-3 bg-orange-50 rounded-lg border border-orange-200">
              <div className="w-4 h-4 bg-orange-500 rounded-full shadow-sm"></div>
              <span className="font-semibold text-gray-700">Decoder Stages (2)</span>
            </div>
            <div className="flex items-center gap-3 p-3 bg-purple-50 rounded-lg border border-purple-200">
              <div className="w-4 h-4 bg-purple-500 rounded-full shadow-sm"></div>
              <span className="font-semibold text-gray-700">Output Layer (1)</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SegFormerExplainer;