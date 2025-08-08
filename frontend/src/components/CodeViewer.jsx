import {useEffect, useState} from 'react';
import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import {oneDark} from 'react-syntax-highlighter/dist/esm/styles/prism';

function CodeViewer({filePath}) {
    const [code, setCode] = useState('');

    useEffect(() => {
        fetch(`http://localhost:8000/file?path=${encodeURIComponent(filePath)}`)
            .then(res => res.text())
            .then(text => setCode(text));
    }, [filePath]);

    return (
        <div style={{height: '100%', overflowY: 'auto'}}>
            <h4>{filePath}</h4>
            <SyntaxHighlighter language="java" style={oneDark} showLineNumbers>
                {code}
            </SyntaxHighlighter>
        </div>
    );
}

export default CodeViewer;