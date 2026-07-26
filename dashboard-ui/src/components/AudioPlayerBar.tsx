import React, { useRef, useState, useEffect } from 'react';
import { Play, Pause, Volume2, VolumeX } from 'lucide-react';

interface AudioPlayerBarProps {
  audioUrl: string;
  onTimeUpdate?: (currentTimeSeconds: number) => void;
  seekTime?: number | null;
}

export const AudioPlayerBar: React.FC<AudioPlayerBarProps> = ({
  audioUrl,
  onTimeUpdate,
  seekTime,
}) => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [isMuted, setIsMuted] = useState(false);

  useEffect(() => {
    if (audioRef.current && typeof seekTime === 'number' && !isNaN(seekTime)) {
      audioRef.current.currentTime = seekTime;
      setCurrentTime(seekTime);
      audioRef.current
        .play()
        .then(() => setIsPlaying(true))
        .catch((err) => console.error('Audio play error on seek:', err));
    }
  }, [seekTime]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current
        .play()
        .then(() => setIsPlaying(true))
        .catch((err) => console.error('Audio play failed:', err));
    }
  };

  const handleTimeUpdate = () => {
    if (!audioRef.current) return;
    const cur = audioRef.current.currentTime;
    setCurrentTime(cur);
    onTimeUpdate?.(cur);
  };

  const handleLoadedMetadata = () => {
    if (!audioRef.current) return;
    setDuration(audioRef.current.duration || 0);
  };

  const handleScrub = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current || !duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const newTime = pct * duration;
    audioRef.current.currentTime = newTime;
    setCurrentTime(newTime);
    onTimeUpdate?.(newTime);
  };

  const cyclePlaybackRate = () => {
    const rates = [1.0, 1.25, 1.5, 2.0];
    const nextRate = rates[(rates.indexOf(playbackRate) + 1) % rates.length];
    setPlaybackRate(nextRate);
    if (audioRef.current) audioRef.current.playbackRate = nextRate;
  };

  const toggleMute = () => {
    if (!audioRef.current) return;
    audioRef.current.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const formatTime = (sec: number) => {
    if (isNaN(sec)) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div
      className="fixed left-0 right-0 bottom-0 z-50 h-[52px] flex items-center gap-3.5 px-5"
      style={{
        background: 'var(--surface-1)',
        borderTop: '1px solid var(--border)',
      }}
    >
      <audio
        ref={audioRef}
        src={audioUrl}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={() => setIsPlaying(false)}
      />

      {/* Play Button */}
      <button
        onClick={togglePlay}
        className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 cursor-pointer border-none"
        style={{ background: 'var(--amber)', color: '#1B1305' }}
        title={isPlaying ? 'Pause' : 'Play'}
      >
        {isPlaying ? (
          <Pause className="w-3 h-3 fill-current" />
        ) : (
          <Play className="w-3 h-3 fill-current ml-px" />
        )}
      </button>

      {/* Time */}
      <span className="num text-[11.5px] flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
        {formatTime(currentTime)}
      </span>

      {/* Scrubber */}
      <div
        className="flex-1 h-[3px] rounded-sm relative cursor-pointer"
        style={{ background: 'var(--surface-3)' }}
        onClick={handleScrub}
      >
        <div
          className="absolute left-0 top-0 bottom-0 rounded-sm"
          style={{ width: `${progress}%`, background: 'var(--amber)' }}
        >
          <div
            className="absolute -right-1 top-1/2 -translate-y-1/2 w-[9px] h-[9px] rounded-full"
            style={{ background: 'var(--amber-strong)' }}
          />
        </div>
      </div>

      {/* Total time */}
      <span className="num text-[11.5px] flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
        {formatTime(duration)}
      </span>

      {/* Controls */}
      <div className="flex items-center gap-3 flex-shrink-0" style={{ color: 'var(--text-tertiary)' }}>
        <button
          onClick={cyclePlaybackRate}
          className="num text-[11px] px-2 py-0.5 rounded-[5px] cursor-pointer border-none"
          style={{ background: 'var(--surface-3)', color: 'var(--text-secondary)' }}
        >
          {playbackRate}x
        </button>

        <button
          onClick={toggleMute}
          className="cursor-pointer border-none bg-transparent p-0"
          style={{ color: isMuted ? 'var(--red-strong)' : 'var(--text-tertiary)' }}
        >
          {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" strokeWidth={1.6} />}
        </button>
      </div>
    </div>
  );
};
