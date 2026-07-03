import React from 'react';
import { Activity } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer, YAxis, Tooltip } from 'recharts';

interface ActivityChartProps {
  data: { time: string; activity: number }[];
  themeRgb: string;
}

export const ActivityChart: React.FC<ActivityChartProps> = ({ data, themeRgb }) => {
  return (
    <div className="bg-black/40 backdrop-blur-md border border-white/20 p-5 w-full shadow-[0_0_30px_rgba(0,0,0,0.8)] mt-4">
      <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-3 flex items-center gap-2">
        <Activity className="w-3 h-3" />
        Activity History
      </h3>
      <div className="h-24 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <Line 
              type="monotone" 
              dataKey="activity" 
              stroke={`rgb(${themeRgb})`} 
              strokeWidth={1} 
              dot={false}
              isAnimationActive={false}
            />
            <YAxis domain={['auto', 'auto']} hide />
            <Tooltip 
              contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.2)', fontSize: '10px' }}
              itemStyle={{ color: `rgb(${themeRgb})` }}
              labelStyle={{ color: 'rgba(255,255,255,0.5)' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
