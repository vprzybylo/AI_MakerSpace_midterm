---
title: GridGuide
emoji: ⚡
colorFrom: blue
colorTo: gray
sdk: streamlit
pinned: false
license: mit
short_description: Field assistant for utility workers
startup_duration_timeout: 1h
---

# Final Submission

1. GitHub: [AI_MakerSpace_midterm](https://github.com/vprzybylo/AI_MakerSpace_midterm)
    1. Video: https://www.loom.com/share/b76593f490554dd0a085fe65d1e2d63a
        
2. Public App link: [GridGuide on Hugging Face](https://huggingface.co/spaces/vanessaprzybylo/gridguide)
3. Public Fine-tuned embeddings: [finetuned_arctic_ft](https://huggingface.co/vanessaprzybylo/finetuned_arctic_ft)

# GridGuide: Field Assistant

A comprehensive tool designed for utility field workers to access Grid Code regulations and real-time weather information, essential for safe and efficient field operations.

## 🎯 Purpose

GridGuide serves two critical functions for field workers:

1. **Grid Code Reference**
   - Quick access to technical specifications and regulations
   - RAG-powered search for accurate information retrieval
   - Context-aware responses for complex regulatory questions

2. **Weather Integration**
   - Real-time weather data from National Weather Service
   - Critical for:
     - Planning outdoor work safely
     - Assessing weather-related risks to equipment
     - Making informed decisions about maintenance schedules
     - Ensuring compliance with weather-dependent regulations

## 🛠️ Technical Stack

- 🤖 **LLM**: GPT-4 
*Chosen for its strong technical understanding and ability to process complex regulatory language*

- 🔢 **Embedding Models**: text-embedding-3-small, BAAI/bge-large-en-v1.5, [finetuned-arctic-ft](https://huggingface.co/vanessaprzybylo/finetuned_arctic_ft)
*Starting with OpenAI for rapid prototyping, then fine-tuning BAAI for domain-specific understanding*

- 🎺 **Orchestration**: LangChain
*Provides robust RAG pipeline components and easy integration with evaluation tools*

- ↗️ **Vector Store**: Qdrant
*Efficient similarity search and excellent support for metadata filtering*

- 📈 **Monitoring**: LangSmith
*Comprehensive tracing and evaluation capabilities for RAG systems*

- 📐 **Evaluation**: RAGAS
*Industry standard for evaluating RAG system performance*

- 💬 **User Interface**: Streamlit
*Fast deployment and simple interface for field workers*

- 🛎️ **Deployment**: Hugging Face Spaces
*Reliable hosting with good uptime and easy deployment*

## 📊 Embedding Model Comparison

### OpenAI text-embedding-3-small
| Metric               | Value   |
|---------------------|---------|
| Faithfulness        | 0.9056  |
| Answer Relevancy    | 0.6007  |
| Context Recall      | 0.7955  |
| Context Precision   | 0.5833  |

### Fine-tuned snowflake-arctic-embed-l
| Metric               | Value   |
|---------------------|---------|
| Faithfulness        | 0.8045  |
| Answer Relevancy    | 0.5738  |
| Context Recall      | 0.5556  |
| Context Precision   | 0.4600  |

### Analysis

1. **OpenAI Superiority**: OpenAI's text-embedding-3-small outperforms the fine-tuned model across all metrics:
   - Higher faithfulness (90.56% vs 80.45%)
   - Better answer relevancy (60.07% vs 57.38%)
   - Stronger context recall (79.55% vs 55.56%)
   - Better context precision (58.33% vs 46.00%)

2. **Fine-tuning Impact**: Despite being fine-tuned on technical documentation, the snowflake-arctic-embed-l model showed:
   - ~10% decrease in faithfulness
   - ~24% decrease in context recall
   - ~12% decrease in context precision

3. **Potential Factors**:
   - The base model may not be as sophisticated as OpenAI's embedding model
   - Fine-tuning dataset might need optimization
   - OpenAI's model may have better pre-training on technical documentation
   - Rate limiting of the API calls to OpenAI may have affected the results

## 📚 Data Strategy

### Data Sources
- [Grid Code PDF](https://www.nationalgrid.com/sites/default/files/documents/8589935310-Complete%20Grid%20Code.pdf)
- National Weather Service API for real-time weather data

### Chunking Strategy
We employ a hybrid chunking approach:
- Section-based chunks for maintaining regulatory context
- Smaller overlapping chunks for detailed technical specifications

**chunk size**: 2000 characters maximum
**chunk_overlap**: 50 characters overlap for context continuity
**separators**: Priority order: double newlines (\n\n), single newline (\n), period (.), space ( ), empty string ("")

## 🔧 Features

- **Unified Interface**: Single input for both weather and Grid Code queries
- **Intelligent Routing**: Automatically determines whether the query is about weather or regulations
- **Weather Information**:
  - Current conditions
  - Temperature and wind data
  - Detailed forecasts
  - Location-specific updates via ZIP code
- **Grid Code Search**:
  - Natural language queries
  - Context-aware responses
  - Relevant section references

## ⚙️ Technical Details

- Built with Streamlit and LangChain
- Uses RAG (Retrieval Augmented Generation) for accurate Grid Code information
- Integrates with National Weather Service API
- Deployment on Hugging Face Spaces
- Extended startup timeout (1h) to accommodate initial PDF processing

## 🔐 Security Note

API keys required:
- OpenAI API key for LLM functionality
- LangChain API key for tracing (optional)

Configure these in Hugging Face Space secrets for deployment.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## 🔮 Future Improvements

### Multi-Modal Enhancements
1. **Voice Input Integration**
   - Enable hands-free operation for field workers
   - Voice-to-text processing for queries while working
   - Support for natural language voice commands
   - Critical for workers wearing safety equipment or working with tools

2. **Image Recognition**
   - Upload photos of equipment for identification
   - Visual inspection assistance
   - Automatic matching of equipment with Grid Code regulations
   - Integration with equipment maintenance documentation

### User Feedback System
1. **Response Quality**
   - Option to mark responses as helpful/unhelpful

2. **Usage Analytics**
   - Track most common query types
   - Identify areas needing improvement
   - Monitor response accuracy over time
   - Analyze user interaction patterns


These enhancements will focus on making the tool more accessible and accurate while gathering valuable user feedback for continuous improvement.
