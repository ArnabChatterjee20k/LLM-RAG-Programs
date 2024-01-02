from langchain.document_loaders import HNLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

loader = HNLoader("https://news.ycombinator.com/")
source = loader.load()
urls = []
for data in source[:3]:
    link = data.metadata["link"]
    urls.append(link)
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

summariser = HuggingFacePipeline.from_model_id(
    model_id="facebook/bart-large-cnn",
    task="summarization",
    # pipeline_kwargs={"max_length": 20, "min_length": 4}
)
result = []
for news in data:
    result.append({"summary":summariser(news.page_content[:1000]),"link":link})
print(result)