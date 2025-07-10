
import React from 'react';
import { LucideIcon } from 'lucide-react';

interface ResultBlockProps {
  icon: LucideIcon;
  iconColor: string;
  label: string;
  value: string | number;
  secondaryLabel?: string;
  secondaryValue?: string | number;
  secondaryColor?: string;
}

const ResultBlock: React.FC<ResultBlockProps> = ({
  icon: Icon,
  iconColor,
  label,
  value,
  secondaryLabel,
  secondaryValue,
  secondaryColor
}) => {
  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-3xl border border-white/10 p-6">
      <p className="flex items-center text-lg font-semibold text-white">
        <Icon className={`w-6 h-6 mr-3 ${iconColor} flex-shrink-0`} />
        {label}: <span className="ml-2 text-white font-bold text-xl">{value}</span>
        {secondaryLabel && secondaryValue && (
          <>
            <span className="mx-4 text-white/40">|</span>
            {secondaryLabel}: <span className={`ml-2 font-bold text-xl ${secondaryColor || 'text-white'}`}>{secondaryValue}</span>
          </>
        )}
      </p>
    </div>
  );
};

export default ResultBlock;
