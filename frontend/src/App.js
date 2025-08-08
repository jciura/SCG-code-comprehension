import ChatPanel from './components/ChatPanel';
import CodeExplorer from './components/CodeExplorer';
import CodeViewer from './components/CodeViewer';
import {useState} from 'react';
import './App.css';

function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [fileContent, setFileContent] = useState('null');

    return (
        <div className="app-container">
            <div className="left-panel">
                <ChatPanel/>
            </div>
            <div className="right-panel">
                <div>
                    <CodeExplorer onFileSelect={setSelectedFile}/>
                </div>
                <div>
                    {selectedFile && (
                        <CodeViewer filePath={selectedFile} setFileContent={setFileContent}/>
                    )}
                </div>
            </div>
        </div>
    );
}

export default App;
