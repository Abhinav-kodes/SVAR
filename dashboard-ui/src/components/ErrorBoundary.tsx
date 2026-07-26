import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error in tab rendering:', error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="glass-card p-8 rounded-2xl text-center space-y-3 my-6">
          <AlertTriangle className="w-10 h-10 text-amber-400 mx-auto" />
          <h4 className="font-display font-bold text-white text-base">Tab Rendering Error</h4>
          <p className="text-slate-400 text-xs max-w-md mx-auto">
            {this.state.error?.message || 'An unexpected rendering issue occurred in this view.'}
          </p>
        </div>
      );
    }

    return this.props.children;
  }
}
