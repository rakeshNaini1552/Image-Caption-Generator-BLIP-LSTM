export default function Caption({ text, loading, model }) {
  if (loading) return <p className="caption-text">Generating caption...</p>
  if (!text) return null
  return (
    <div className="caption-result">
      <p className="caption-text">"{text}"</p>
      <span className="caption-model">{model?.toUpperCase()}</span>
    </div>
  )
}
