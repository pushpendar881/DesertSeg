import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const image = formData.get('image') as File

    if (!image) {
      return NextResponse.json(
        { error: 'No image provided' },
        { status: 400 }
      )
    }

    // Pass raw image file directly to backend - NO base64 conversion (preserves quality)
    const backendFormData = new FormData()
    backendFormData.append('image', image)

    console.log('[v0] Forwarding raw image to backend for parallel prediction:', image.name)

    // Backend API URL - use environment variable or default to localhost
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000'
    
    try {
      // Forward raw multipart/form-data to backend - image goes directly, no quality loss
      const backendResponse = await fetch(`${backendUrl}/predict-parallel`, {
        method: 'POST',
        body: backendFormData,
        // Don't set Content-Type - fetch sets multipart boundary automatically
      })

      if (!backendResponse.ok) {
        const errorText = await backendResponse.text()
        throw new Error(`Backend API error: ${backendResponse.status} - ${errorText}`)
      }

      const result = await backendResponse.json()
      console.log('[v0] Parallel prediction successful from backend')
      console.log('[v0] Response preview:', {
        success: result.success,
        modelsRun: result.metadata?.modelsRun,
        results: Object.keys(result.results || {}),
      })
      return NextResponse.json(result)
    } catch (backendError) {
      console.error('[v0] Backend API error:', backendError)
      // Return error response with zeros/empty values instead of mock data
      return NextResponse.json(
        {
          success: false,
          error: 'Backend unavailable',
          results: {},
          comparativeAnalysis: {
            summary: 'Model prediction unavailable. Please ensure the backend server is running.',
            classComparison: [],
            processingTimeComparison: {
              times: {},
              fastest: null,
              slowest: null,
            },
            modelInsights: [],
          },
          metadata: {
            totalProcessingTime: 0,
            imageSize: { width: 0, height: 0 },
            modelsRun: 0,
          },
        },
        { status: 503 }
      )
    }
  } catch (error) {
    console.error('[v0] Parallel prediction error:', error)
    return NextResponse.json(
      { error: 'Failed to process image', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
}
