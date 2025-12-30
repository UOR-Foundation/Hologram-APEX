import { useModels } from './hooks/useModels';
import { useTheme } from './hooks/useTheme';
import { Header } from './components/Header';
import { ChatContainer } from './components/ChatContainer';
import './index.css';

function App() {
  const { theme, toggleTheme } = useTheme();
  const { models, selectedModel, setSelectedModel, isLoading } = useModels();

  return (
    <div className="flex h-screen flex-col bg-surface text-foreground">
      <Header
        models={models}
        selectedModel={selectedModel}
        onModelSelect={setSelectedModel}
        isLoading={isLoading}
        onThemeToggle={toggleTheme}
        theme={theme}
      />
      <main className="flex-1 overflow-hidden">
        <ChatContainer selectedModel={selectedModel} />
      </main>
    </div>
  );
}

export default App;
