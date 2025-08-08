import {useEffect, useState} from "react";

function CodeExplorer({onFileSelect}) {
    const [files, setFiles] = useState([]);

    useEffect(() => {
        fetch("http://127.0.0.1:8000/files?path=test%2Fsrc%2Fmain%2Fjava%2Fcom%2Fto%2Fproj%2FUser")
            .then(res => res.json())
            .then(data => {
                console.log(data);
                setFiles(data);
            })
    }, []);

    return (
        <div>
            <h4>Pliki</h4>
            <ul>
                {files.map((file) => (
                    <li key={file} onClick={() => onFileSelect("test/src/main/java/com/to/proj/User/" + file)}>
                        {file}
                    </li>
                ))}
            </ul>
        </div>
    )
}

export default CodeExplorer;