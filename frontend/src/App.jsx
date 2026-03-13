import { useState } from 'react'
import './App.css'
import Caption from './components/Caption'
import ImageUploader from './components/ImageUploader'

export default function App() {
  const [preview, setPreview] = useState(null)
  const [file, setFile] = useState(null)
  const [caption, setCaption] = useState('')
  const [captionModel, setCaptionModel] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [modelType, setModelType] = useState('blip')

  function handleFileSelect(selected) {
    setFile(selected)
    setCaption('')
    setError('')
    setPreview(URL.createObjectURL(selected))
  }

  async function handleSubmit() {
    if (!file) return
    setLoading(true)
    setCaption('')
    setError('')

    const form = new FormData()
    form.append('file', file)
    form.append('model_type', modelType)

    try {
      const res = await fetch('http://localhost:8000/caption', {
        method: 'POST',
        body: form,
      })
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data = await res.json()
      setCaption(data.caption)
      setCaptionModel(data.model)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <h1>Image Captioner</h1>
      <ImageUploader onFileSelect={handleFileSelect} />
      <div className="model-toggle">
        {['blip', 'lstm'].map((type) => (
          <button
            key={type}
            className={`toggle-btn ${modelType === type ? 'active' : ''}`}
            onClick={() => setModelType(type)}
          >
            {type.toUpperCase()}
          </button>
        ))}
      </div>
      {preview && (
        <img className="preview" src={preview} alt="Preview" />
      )}
      {file && !loading && (
        <button className="caption-btn" onClick={handleSubmit}>
          Generate Caption
        </button>
      )}
      <Caption text={caption} loading={loading} model={captionModel} />
      {error && <p className="error">{error}</p>}
    </div>
  )
}
