'use client';

import dynamic from 'next/dynamic';

// Dynamically import the SegFormer explainer to avoid SSR issues
const SegFormerExplainer = dynamic(() => import('@/components/segformer-explainer'), {
  ssr: false,
  loading: () => (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-center">
        <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-muted-foreground">Loading SegFormer Architecture Explainer...</p>
      </div>
    </div>
  )
});

export default function ExplainerPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto py-8">
        <SegFormerExplainer />
      </div>
    </div>
  );
}