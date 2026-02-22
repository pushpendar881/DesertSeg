'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

const segmentationClasses = [
  { name: 'Trees', color: '#2d6a4f' },
  { name: 'Lush Bushes', color: '#52b788' },
  { name: 'Dry Grass', color: '#d4a017' },
  { name: 'Dry Bushes', color: '#a07850' },
  { name: 'Ground Clutter', color: '#6b6b6b' },
  { name: 'Flowers', color: '#e76f51' },
  { name: 'Logs', color: '#8b5e3c' },
  { name: 'Rocks', color: '#9e9e9e' },
  { name: 'Landscape', color: '#c8a96e' },
  { name: 'Sky', color: '#90c8e0' },
]

export function HowItWorksSection() {
  return (
    <section id="how-it-works" className="py-24 bg-zinc-900">
      <div className="container mx-auto px-6 max-w-7xl">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Understanding the Models
          </h2>
          <p className="text-lg text-gray-400 max-w-3xl mx-auto">
            A guided walkthrough of how SegFormer-10-Classes-50-Epochs and DeepLabV3+ segment desert scenes
          </p>
        </div>

        {/* What is Semantic Segmentation */}
        <Card className="mb-16">
          <CardHeader>
            <CardTitle className="text-2xl">What is Semantic Segmentation?</CardTitle>
            <CardDescription>
              Assigning a class label to every pixel in an image
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col md:flex-row items-center gap-8 mb-8">
              <div className="flex-1 space-y-4">
                <div className="text-center">
                  <p className="text-sm font-medium mb-2">Input Image</p>
                  <div className="w-full h-32 bg-gradient-to-b from-sky-300 to-yellow-200 rounded-lg border border-border"></div>
                </div>
              </div>
              <div className="text-4xl text-muted-foreground">→</div>
              <div className="flex-1 space-y-4">
                <div className="text-center">
                  <p className="text-sm font-medium mb-2">Model Processing</p>
                  <div className="w-full h-32 bg-primary/10 rounded-lg border border-primary/20 flex items-center justify-center">
                    <p className="text-sm text-primary font-semibold">Neural Network</p>
                  </div>
                </div>
              </div>
              <div className="text-4xl text-muted-foreground">→</div>
              <div className="flex-1 space-y-4">
                <div className="text-center">
                  <p className="text-sm font-medium mb-2">Segmentation Mask</p>
                  <div className="w-full h-32 rounded-lg border border-border overflow-hidden grid grid-cols-3">
                    {[...Array(9)].map((_, i) => (
                      <div
                        key={i}
                        style={{ backgroundColor: segmentationClasses[i]?.color }}
                        className="w-full h-full"
                      />
                    ))}
                  </div>
                </div>
              </div>
            </div>
            <div className="flex flex-wrap gap-2 justify-center">
              {segmentationClasses.map((cls) => (
                <div key={cls.name} className="flex items-center gap-2">
                  <div
                    className="w-4 h-4 rounded"
                    style={{ backgroundColor: cls.color }}
                  />
                  <span className="text-xs text-muted-foreground">{cls.name}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* SegFormer-10-Classes-50-Epochs Explainer */}
        <div className="mb-16">
          <div className="mb-8">
            <Badge className="mb-4 bg-accent text-white">SegFormer-10-Classes-50-Epochs</Badge>
            <h3 className="text-3xl font-bold text-white mb-2">
              SegFormer-10-Classes-50-Epochs: Transformer-Based Segmentation
            </h3>
            <p className="text-gray-400">
              A hierarchical transformer encoder paired with a lightweight MLP decoder
            </p>
          </div>

          <div className="space-y-6">
            {/* Step 1 */}
            <Card>
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold">
                    1
                  </div>
                  <CardTitle>Patch Embedding</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <p className="text-muted-foreground leading-relaxed">
                      The input image is divided into overlapping 4×4 patches arranged in a grid.
                      Unlike ViT's 16×16 patches, smaller patches preserve fine spatial detail
                      critical for dense prediction.
                    </p>
                  </div>
                  <div className="bg-muted/50 rounded-lg p-6 flex items-center justify-center">
                    <div className="grid grid-cols-8 gap-1">
                      {[...Array(64)].map((_, i) => (
                        <div
                          key={i}
                          className="w-6 h-6 bg-primary/20 border border-primary/40 rounded"
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Step 2 */}
            <Card>
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold">
                    2
                  </div>
                  <CardTitle>Hierarchical Encoder (4 Stages)</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <p className="text-muted-foreground leading-relaxed">
                      Four transformer stages progressively reduce resolution while increasing
                      channel depth — producing both fine local features and broad global context.
                    </p>
                  </div>
                  <div className="bg-muted/50 rounded-lg p-6 flex items-center justify-center">
                    <div className="flex items-end gap-4">
                      {[
                        { h: 80, label: 'H/4' },
                        { h: 60, label: 'H/8' },
                        { h: 40, label: 'H/16' },
                        { h: 20, label: 'H/32' },
                      ].map((stage, i) => (
                        <div key={i} className="flex flex-col items-center gap-2">
                          <div
                            className="w-16 bg-primary/30 border-2 border-primary rounded"
                            style={{ height: `${stage.h}px` }}
                          />
                          <span className="text-xs text-muted-foreground font-mono">{stage.label}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Step 3 */}
            <Card>
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold">
                    3
                  </div>
                  <CardTitle>Efficient Self-Attention</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <p className="text-muted-foreground leading-relaxed">
                      Standard self-attention is O(N²). SegFormer-10-Classes-50-Epochs reduces K and V by ratio R,
                      making attention efficient enough for high-resolution feature maps.
                    </p>
                  </div>
                  <div className="bg-muted/50 rounded-lg p-6 flex items-center justify-center">
                    <div className="space-y-3">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-primary"></div>
                        <div className="flex-1 h-px bg-primary/30"></div>
                        <div className="w-3 h-3 rounded-full bg-primary/60"></div>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-primary"></div>
                        <div className="flex-1 h-px bg-primary/30"></div>
                        <div className="w-3 h-3 rounded-full bg-primary/60"></div>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-primary"></div>
                        <div className="flex-1 h-px bg-primary/30"></div>
                        <div className="w-3 h-3 rounded-full bg-primary/60"></div>
                      </div>
                      <p className="text-xs text-center text-muted-foreground mt-4">Reduction ratio R</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Step 4 */}
            <Card>
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold">
                    4
                  </div>
                  <CardTitle>Mix-FFN (No Positional Encoding)</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <p className="text-muted-foreground leading-relaxed">
                      Instead of fixed positional encodings (which fail at new resolutions), SegFormer-10-Classes-50-Epochs
                      uses a 3×3 convolution inside the FFN to implicitly encode position — enabling
                      zero-shot resolution generalization.
                    </p>
                  </div>
                  <div className="bg-muted/50 rounded-lg p-6 flex items-center justify-center">
                    <div className="flex items-center gap-3">
                      <div className="px-4 py-2 bg-primary/20 border border-primary rounded text-sm">Input</div>
                      <div className="text-muted-foreground">→</div>
                      <div className="px-4 py-2 bg-primary/20 border border-primary rounded text-sm">MLP</div>
                      <div className="text-muted-foreground">→</div>
                      <div className="px-4 py-2 bg-primary border border-primary rounded text-sm text-primary-foreground">Conv3×3</div>
                      <div className="text-muted-foreground">→</div>
                      <div className="px-4 py-2 bg-primary/20 border border-primary rounded text-sm">GELU</div>
                      <div className="text-muted-foreground">→</div>
                      <div className="px-4 py-2 bg-primary/20 border border-primary rounded text-sm">Output</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Step 5 */}
            <Card>
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold">
                    5
                  </div>
                  <CardTitle>All-MLP Decoder</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <p className="text-muted-foreground leading-relaxed">
                      Multi-scale features are unified by channel-wise MLPs, upsampled to 1/4
                      resolution, concatenated, and fused by a final MLP. No complex CNN modules needed.
                    </p>
                  </div>
                  <div className="bg-muted/50 rounded-lg p-6 flex items-center justify-center">
                    <div className="flex flex-col items-center gap-3">
                      <div className="flex gap-2">
                        {[1, 2, 3, 4].map((i) => (
                          <div key={i} className="px-3 py-2 bg-primary/20 border border-primary rounded text-xs">
                            Stage {i}
                          </div>
                        ))}
                      </div>
                      <div className="text-muted-foreground">↓ Upsample & Concat</div>
                      <div className="px-6 py-3 bg-primary border border-primary rounded text-sm text-primary-foreground">
                        Final MLP
                      </div>
                      <div className="text-muted-foreground">↓</div>
                      <div className="px-6 py-3 bg-accent/20 border border-accent rounded text-sm">
                        Segmentation Mask
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* DeepLabV3+ Explainer */}
        <div className="mb-16">
          <div className="mb-8">
            <Badge className="mb-4" style={{ backgroundColor: '#88E7B2', color: '#0A0A0A' }}>DeepLabV3+</Badge>
            <h3 className="text-3xl font-bold text-white mb-2">
              DeepLabV3+: Atrous Convolution Segmentation
            </h3>
            <p className="text-gray-400">
              CNN-based architecture with atrous spatial pyramid pooling
            </p>
          </div>

          <div className="space-y-6">
            {/* Step 1 */}
            <Card>
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-8 h-8 rounded-full bg-accent text-accent-foreground flex items-center justify-center font-bold">
                    1
                  </div>
                  <CardTitle>Backbone Encoding</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <p className="text-muted-foreground leading-relaxed">
                      ResNet-101 extracts deep feature representations from the input image through
                      residual convolutional blocks.
                    </p>
                  </div>
                  <div className="bg-muted/50 rounded-lg p-6 flex items-center justify-center">
                    <div className="flex items-center gap-2">
                      {[64, 128, 256, 512, 1024].map((channels, i) => (
                        <div key={i} className="flex flex-col items-center">
                          <div
                            className="w-12 bg-accent/30 border-2 border-accent rounded"
                            style={{ height: `${100 - i * 15}px` }}
                          />
                          <span className="text-xs text-muted-foreground mt-1">{channels}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Step 2 */}
            <Card>
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-8 h-8 rounded-full bg-accent text-accent-foreground flex items-center justify-center font-bold">
                    2
                  </div>
                  <CardTitle>Atrous (Dilated) Convolutions</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <p className="text-muted-foreground leading-relaxed">
                      Atrous convolutions inflate the convolution kernel with holes, expanding the
                      receptive field without increasing parameters or losing resolution.
                    </p>
                  </div>
                  <div className="bg-muted/50 rounded-lg p-6 flex items-center justify-center gap-8">
                    <div className="flex flex-col items-center gap-2">
                      <div className="grid grid-cols-3 gap-1">
                        {[...Array(9)].map((_, i) => (
                          <div key={i} className="w-4 h-4 bg-accent border border-accent rounded" />
                        ))}
                      </div>
                      <span className="text-xs text-muted-foreground">Rate=1</span>
                    </div>
                    <div className="flex flex-col items-center gap-2">
                      <div className="grid grid-cols-5 gap-1">
                        {[...Array(25)].map((_, i) => {
                          const isActive = [0, 2, 4, 10, 12, 14, 20, 22, 24].includes(i)
                          return (
                            <div
                              key={i}
                              className={`w-4 h-4 border rounded ${isActive
                                  ? 'bg-accent border-accent'
                                  : 'bg-transparent border-accent/20 border-dashed'
                                }`}
                            />
                          )
                        })}
                      </div>
                      <span className="text-xs text-muted-foreground">Rate=2</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Step 3 */}
            <Card>
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-8 h-8 rounded-full bg-accent text-accent-foreground flex items-center justify-center font-bold">
                    3
                  </div>
                  <CardTitle>ASPP Module (Atrous Spatial Pyramid Pooling)</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <p className="text-muted-foreground leading-relaxed">
                      ASPP captures multi-scale context by applying parallel atrous convolutions at
                      multiple rates, then fusing them. This gives the model awareness of objects at
                      different scales.
                    </p>
                  </div>
                  <div className="bg-muted/50 rounded-lg p-6 flex items-center justify-center">
                    <div className="flex flex-col items-center gap-3">
                      <div className="px-6 py-3 bg-accent/20 border border-accent rounded">Input</div>
                      <div className="flex gap-2">
                        {['1×1', 'Rate 6', 'Rate 12', 'Rate 18', 'Pool'].map((label) => (
                          <div key={label} className="px-2 py-2 bg-accent/30 border border-accent rounded text-xs">
                            {label}
                          </div>
                        ))}
                      </div>
                      <div className="text-muted-foreground text-sm">↓ Concatenate</div>
                      <div className="px-6 py-3 bg-accent border border-accent rounded text-accent-foreground">
                        Fused Features
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Step 4 */}
            <Card>
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-8 h-8 rounded-full bg-accent text-accent-foreground flex items-center justify-center font-bold">
                    4
                  </div>
                  <CardTitle>Encoder–Decoder Structure</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <p className="text-muted-foreground leading-relaxed">
                      DeepLabV3+ combines rich semantic features from the encoder with fine spatial
                      details from early layers, recovering sharp boundaries through the decoder.
                    </p>
                  </div>
                  <div className="bg-muted/50 rounded-lg p-6 flex items-center justify-center">
                    <div className="flex flex-col items-center gap-3">
                      <div className="flex gap-4">
                        <div className="flex flex-col items-center gap-2">
                          <div className="px-4 py-3 bg-accent/30 border border-accent rounded text-xs">
                            Encoder
                            <br />
                            (low-res, rich)
                          </div>
                        </div>
                        <div className="flex items-center text-2xl text-muted-foreground">+</div>
                        <div className="flex flex-col items-center gap-2">
                          <div className="px-4 py-3 bg-accent/20 border border-accent/50 rounded text-xs">
                            Low-level
                            <br />
                            (high-res, shallow)
                          </div>
                        </div>
                      </div>
                      <div className="text-muted-foreground text-sm">↓ Upsample & Fuse</div>
                      <div className="px-6 py-3 bg-accent border border-accent rounded text-accent-foreground">
                        Segmentation Mask
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Head-to-Head Comparison */}
        <Card>
          <CardHeader>
            <CardTitle className="text-2xl">Head-to-Head Concept Comparison</CardTitle>
            <CardDescription>Key architectural differences between the two approaches</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-3 px-4 font-medium">Property</th>
                    <th className="text-left py-3 px-4 font-medium text-primary">SegFormer-10-Classes-50-Epochs</th>
                    <th className="text-left py-3 px-4 font-medium text-accent">DeepLabV3+</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-border/50 bg-muted/20">
                    <td className="py-3 px-4 font-medium">Core mechanism</td>
                    <td className="py-3 px-4 text-primary">Self-attention</td>
                    <td className="py-3 px-4 text-accent">Atrous convolution</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-medium">Positional encoding</td>
                    <td className="py-3 px-4 text-primary">None (Mix-FFN)</td>
                    <td className="py-3 px-4 text-accent">Implicit via conv</td>
                  </tr>
                  <tr className="border-b border-border/50 bg-muted/20">
                    <td className="py-3 px-4 font-medium">Multi-scale features</td>
                    <td className="py-3 px-4 text-primary">Hierarchical encoder</td>
                    <td className="py-3 px-4 text-accent">ASPP module</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-medium">Decoder complexity</td>
                    <td className="py-3 px-4 text-primary">Lightweight MLP</td>
                    <td className="py-3 px-4 text-accent">Heavier CNN decoder</td>
                  </tr>
                  <tr className="border-b border-border/50 bg-muted/20">
                    <td className="py-3 px-4 font-medium">Resolution flexibility</td>
                    <td className="py-3 px-4 text-primary">High (no fixed PE)</td>
                    <td className="py-3 px-4 text-accent">Moderate</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-medium">Best for</td>
                    <td className="py-3 px-4 text-primary">Varied resolutions</td>
                    <td className="py-3 px-4 text-accent">Established datasets</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>
    </section>
  )
}
