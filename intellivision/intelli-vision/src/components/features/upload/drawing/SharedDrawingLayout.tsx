import React from "react";

interface SharedDrawingLayoutProps {
  title: string;
  description: string;
  statusCard?: React.ReactNode;
  controls: React.ReactNode;
  listSection?: React.ReactNode;
  preview: React.ReactNode;
  footer?: React.ReactNode;
}

const SharedDrawingLayout: React.FC<SharedDrawingLayoutProps> = ({
  title,
  description,
  statusCard,
  controls,
  listSection,
  preview,
  footer,
}) => (
  <div className="w-full space-y-8">
    {/* Header Section */}
    <div className="text-center space-y-4">
      <h2 className="text-3xl font-bold text-white">{title}</h2>
      <p className="text-white/70 text-lg max-w-3xl mx-auto leading-relaxed">
        {description}
      </p>
    </div>

    {/* Main Content - Two Column Layout */}
    <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
      {/* Left Column: Controls & Status */}
      <div className="xl:col-span-2 space-y-6">
        <div className="card-base rounded-3xl h-fit">
          {/* Status Card */}
          {statusCard && <div className="mb-6">{statusCard}</div>}

          {/* Controls */}
          <div className="space-y-6">{controls}</div>
        </div>

        {/* List Section */}
        {listSection && (
          <div className="glass-card rounded-3xl border-white/10 p-6">
            {listSection}
          </div>
        )}
      </div>

      {/* Right Column: Preview */}
      <div className="xl:col-span-3">
        <div className="h-full flex flex-col">{preview}</div>
      </div>
    </div>

    {/* Footer - Full width below grid */}
    {footer && <div className="w-full">{footer}</div>}
  </div>
);

export default SharedDrawingLayout;
