import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const video = formData.get('video') as File
    const model = (formData.get('model') as string) || 'segformer'
    const fpsLimit = (formData.get('fps_limit') as string) || '5'
    const overlayAlpha = (formData.get('overlay_alpha') as string) || '0.6'

    if (!video) {
      return NextResponse.json(
        { error: 'No video provided' },
        { status: 400 }
      )
    }

    const backendFormData = new FormData()
    backendFormData.append('video', video)
    backendFormData.append('model', model)
    backendFormData.append('fps_limit', fpsLimit)
    backendFormData.append('overlay_alpha', overlayAlpha)

    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000'

    try {
      const backendResponse = await fetch(`${backendUrl}/predict-video`, {
        method: 'POST',
        body: backendFormData,
      })

      if (!backendResponse.ok) {
        const errorText = await backendResponse.text()
        throw new Error(`Backend API error: ${backendResponse.status} - ${errorText}`)
      }

      const result = await backendResponse.json()
      return NextResponse.json(result)
    } catch (backendError) {
      console.error('[video] Backend API error:', backendError)
      return NextResponse.json(
        {
          success: false,
          error: 'Backend unavailable',
          video: null,
          metadata: {
            processingTime: 0,
            framesTotal: 0,
            framesProcessed: 0,
            originalFps: 0,
            resolution: { width: 0, height: 0 },
            modelUsed: model,
          },
        },
        { status: 503 }
      )
    }
  } catch (error) {
    console.error('[video] Prediction error:', error)
    return NextResponse.json(
      {
        error: 'Failed to process video',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}

