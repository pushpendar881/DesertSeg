// import { NextRequest, NextResponse } from 'next/server'

// export async function POST(request: NextRequest) {
//   try {
//     const formData = await request.formData()
//     const video = formData.get('video') as File
//     const model = (formData.get('model') as string) || 'segformer'
//     const fpsLimit = (formData.get('fps_limit') as string) || '5'
//     const overlayAlpha = (formData.get('overlay_alpha') as string) || '0.6'

//     if (!video) {
//       return NextResponse.json(
//         { error: 'No video provided' },
//         { status: 400 }
//       )
//     }

//     const backendFormData = new FormData()
//     backendFormData.append('video', video)
//     backendFormData.append('model', model)
//     backendFormData.append('fps_limit', fpsLimit)
//     backendFormData.append('overlay_alpha', overlayAlpha)

//     // Prefer explicit IPv4 loopback to avoid any localhost/IPv6 quirks
//     const backendUrl = process.env.BACKEND_URL || 'http://127.0.0.1:8000'
//     console.log('[video] Using backend URL:', backendUrl)

//     try {
//       const backendResponse = await fetch(`${backendUrl}/predict-video`, {
//         method: 'POST',
//         body: backendFormData,
//       })

//       if (!backendResponse.ok) {
//         const errorText = await backendResponse.text()
//         throw new Error(`Backend API error: ${backendResponse.status} - ${errorText}`)
//       }

//       // Backend now returns video file directly, not JSON
//       // Extract metadata from response headers
//       const videoBlob = await backendResponse.blob()
//       const metadata = {
//         processingTime: parseFloat(backendResponse.headers.get('X-Processing-Time') || '0'),
//         framesTotal: parseInt(backendResponse.headers.get('X-Frames-Total') || '0', 10),
//         framesProcessed: parseInt(backendResponse.headers.get('X-Frames-Processed') || '0', 10),
//         originalFps: parseFloat(backendResponse.headers.get('X-Original-FPS') || '0'),
//         resolution: (() => {
//           const res = backendResponse.headers.get('X-Resolution') || '0x0'
//           const [width, height] = res.split('x').map(Number)
//           return { width, height }
//         })(),
//         modelUsed: backendResponse.headers.get('X-Model-Used') || model,
//       }

//       // Create blob URL for the video
//       const videoUrl = URL.createObjectURL(videoBlob)

//       // Return JSON with blob URL and metadata
//       return NextResponse.json({
//         success: true,
//         video: videoUrl, // Blob URL instead of base64
//         metadata,
//       })
//     } catch (backendError) {
//       console.error('[video] Backend API error:', backendError)
//       return NextResponse.json(
//         {
//           success: false,
//           error: 'Backend unavailable',
//           details:
//             backendError instanceof Error
//               ? backendError.message
//               : String(backendError),
//           video: null,
//           metadata: {
//             processingTime: 0,
//             framesTotal: 0,
//             framesProcessed: 0,
//             originalFps: 0,
//             resolution: { width: 0, height: 0 },
//             modelUsed: model,
//           },
//         },
//         { status: 503 }
//       )
//     }
//   } catch (error) {
//     console.error('[video] Prediction error:', error)
//     return NextResponse.json(
//       {
//         error: 'Failed to process video',
//         details: error instanceof Error ? error.message : 'Unknown error',
//       },
//       { status: 500 }
//     )
//   }
// }


import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const video = formData.get('video') as File
    const model = (formData.get('model') as string) || 'segformer'
    const fpsLimit = (formData.get('fps_limit') as string) || '5'
    const overlayAlpha = (formData.get('overlay_alpha') as string) || '0.6'

    if (!video) {
      return NextResponse.json({ error: 'No video provided' }, { status: 400 })
    }

    const backendFormData = new FormData()
    backendFormData.append('video', video)
    backendFormData.append('model', model)
    backendFormData.append('fps_limit', fpsLimit)
    backendFormData.append('overlay_alpha', overlayAlpha)

    const backendUrl = process.env.BACKEND_URL || 'http://127.0.0.1:8000'
    console.log('[video] Using backend URL:', backendUrl)

    const backendResponse = await fetch(`${backendUrl}/predict-video`, {
      method: 'POST',
      body: backendFormData,
    })

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text()
      throw new Error(`Backend error: ${backendResponse.status} - ${errorText}`)
    }

    // Read video bytes on server side and convert to base64
    // (URL.createObjectURL is browser-only and cannot be used here)
    const videoBytes = await backendResponse.arrayBuffer()
    const base64Video = Buffer.from(videoBytes).toString('base64')

    // Extract metadata from response headers
    const metadata = {
      processingTime:  parseFloat(backendResponse.headers.get('X-Processing-Time') || '0'),
      framesTotal:     parseInt(backendResponse.headers.get('X-Frames-Total') || '0', 10),
      framesProcessed: parseInt(backendResponse.headers.get('X-Frames-Processed') || '0', 10),
      originalFps:     parseFloat(backendResponse.headers.get('X-Original-FPS') || '0'),
      resolution: (() => {
        const res = backendResponse.headers.get('X-Resolution') || '0x0'
        const [width, height] = res.split('x').map(Number)
        return { width, height }
      })(),
      modelUsed: backendResponse.headers.get('X-Model-Used') || model,
    }

    return NextResponse.json({
      success: true,
      // Send as data URI — the frontend will set this directly as <video src>
      video: `data:video/mp4;base64,${base64Video}`,
      metadata,
    })

  } catch (error) {
    console.error('[video] Prediction error:', error)
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to process video',
        details: error instanceof Error ? error.message : 'Unknown error',
        video: null,
        metadata: {
          processingTime: 0, framesTotal: 0, framesProcessed: 0,
          originalFps: 0, resolution: { width: 0, height: 0 }, modelUsed: '',
        },
      },
      { status: 500 }
    )
  }
}

