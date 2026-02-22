import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const image = formData.get('image') as File
    const model = formData.get('model') as string

    if (!image) {
      return NextResponse.json(
        { error: 'No image provided' },
        { status: 400 }
      )
    }

    // Pass raw image file directly to backend - NO base64 conversion (preserves quality)
    const backendFormData = new FormData()
    backendFormData.append('image', image)
    backendFormData.append('model', model || 'segformer')

    console.log('[v0] Forwarding raw image to backend:', image.name, 'Model:', model)

    // Backend API URL - use environment variable or default to localhost
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000'
    
    try {
      // Forward raw multipart/form-data to backend - image goes directly, no quality loss
      const backendResponse = await fetch(`${backendUrl}/predict`, {
        method: 'POST',
        body: backendFormData,
        // Don't set Content-Type - fetch sets multipart boundary automatically
      })

      if (!backendResponse.ok) {
        const errorText = await backendResponse.text()
        throw new Error(`Backend API error: ${backendResponse.status} - ${errorText}`)
      }

      const result = await backendResponse.json()
      console.log('[v0] Prediction successful from backend')
      console.log('[v0] Response preview:', {
        success: result.success,
        hasMask: !!result.segmentationMask,
        maskLength: result.segmentationMask?.length,
      })
      return NextResponse.json(result)
    } catch (backendError) {
      console.error('[v0] Backend API error:', backendError)
      // Return error response with zeros/empty values instead of mock data
      return NextResponse.json(
        {
          success: false,
          error: 'Backend unavailable',
          segmentationMask: `data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==`,
          reasoning: {
            explanation: 'Model prediction unavailable. Please ensure the backend server is running.',
            attentionMap: `data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==`,
            featureImportance: [
              { feature: 'Texture patterns', confidence: 0 },
              { feature: 'Color distribution', confidence: 0 },
              { feature: 'Edge boundaries', confidence: 0 },
              { feature: 'Spatial context', confidence: 0 },
              { feature: 'Shape geometry', confidence: 0 },
            ],
            classConfidence: [
              { class: 'Trees', confidence: 0 },
              { class: 'Lush Bushes', confidence: 0 },
              { class: 'Dry Grass', confidence: 0 },
              { class: 'Dry Bushes', confidence: 0 },
              { class: 'Ground Clutter', confidence: 0 },
              { class: 'Flowers', confidence: 0 },
              { class: 'Logs', confidence: 0 },
              { class: 'Rocks', confidence: 0 },
              { class: 'Landscape', confidence: 0 },
              { class: 'Sky', confidence: 0 },
            ],
          },
          metadata: {
            processingTime: 0,
            modelVersion: '0.0.0',
            imageSize: { width: 0, height: 0 },
          },
        },
        { status: 503 }
      )
    }
  } catch (error) {
    console.error('[v0] Prediction error:', error)
    return NextResponse.json(
      { error: 'Failed to process image', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
}
