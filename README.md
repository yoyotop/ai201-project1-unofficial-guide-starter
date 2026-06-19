# The Unofficial Guide — CSI Computer Science Professor Reviews

A Retrieval-Augmented Generation (RAG) system that helps students find honest, practical feedback about College of Staten Island (CSI) Computer Science professors and courses.

---

## Domain

This project focuses on student-generated reviews and discussions about College of Staten Island (CSI) Computer Science courses and professors.

While CSI provides official course descriptions, faculty profiles, and curriculum information, students often want practical insights about teaching styles, workload, grading policies, exam difficulty, communication, and classroom experiences. This information is rarely available through official channels and is typically scattered across review websites, discussion boards, and student conversations.

The goal of this project is to make that information searchable through natural language questions using a Retrieval-Augmented Generation (RAG) pipeline.

---

## Document Sources

| #  | Source                               | Type      | URL or file path                           |
| -- | ------------------------------------ | --------- | ------------------------------------------ |
| 1  | Professor Ali Mohamed Reviews        | Text File | documents/professor_ali_mohamed.txt        |
| 2  | Professor Joseph Frusci Reviews      | Text File | documents/professor_joseph_frusci.txt      |
| 3  | Professor Jun Rao Reviews            | Text File | documents/professor_jun_rao.txt            |
| 4  | Professor Louis Petingi Reviews      | Text File | documents/professor_louis_petingi.txt      |
| 5  | Professor Peng Shi Reviews           | Text File | documents/professor_peng_shi.txt           |
| 6  | Professor Shuqun Zhang Reviews       | Text File | documents/professor_shuqun_zhang.txt       |
| 7  | Professor Susan Imberman Reviews     | Text File | documents/professor_susan_imberman.txt     |
| 8  | Professor Tatiana Anderson Reviews   | Text File | documents/professor_tatiana_anderson.txt   |
| 9  | Professor Yu Mei Huo Reviews         | Text File | documents/professor_yu_mei_huo.txt         |
| 10 | Professor Zaid Al Mashhadani Reviews | Text File | documents/professor_zaid_al_mashhadani.txt |

These documents contain collected student opinions, experiences, and recommendations regarding CSI Computer Science faculty.

---

## Chunking Strategy

**Chunk size:** 400 characters

**Overlap:** 50 characters

**Why these choices fit your documents:**

Most professor reviews are relatively short and focused on a specific topic such as teaching style, exams, assignments, workload, or communication. Using a chunk size of 400 characters preserves enough context for semantic retrieval while keeping chunks focused on a single idea. A 50-character overlap helps prevent important information from being lost when a review spans multiple chunks.

**Final chunk count:**

85 chunks

---

## Embedding Model

**Model used:**

`all-MiniLM-L6-v2`

This embedding model was accessed through the Sentence Transformers library.

**Production tradeoff reflection:**

The all-MiniLM-L6-v2 model provides strong semantic search performance while remaining lightweight and fast enough to run locally. It is a good choice for a student project because it requires no external embedding API and produces high-quality vector representations.

If cost were not a constraint, I would consider larger embedding models with stronger semantic understanding and multilingual support. Larger models could improve retrieval accuracy, especially when users ask questions using wording that differs significantly from the original reviews. The tradeoff would be increased latency, storage requirements, and computational cost.

---

## Grounded Generation

**System prompt grounding instruction:**

The language model is instructed to answer questions using only the retrieved document context. If the retrieved documents do not contain enough information to answer a question, the model is instructed to state that it does not have sufficient information rather than generating unsupported claims.

The retrieval pipeline first finds the most relevant chunks using semantic search and then passes only those chunks into the prompt used for generation.

**How source attribution is surfaced in the response:**

Each retrieved chunk retains metadata identifying the source document. After generating a response, the system displays the source files used to answer the question. This allows users to verify where information originated and helps maintain transparency.

Example:

```
Sources:
professor_yu_mei_huo.txt
professor_shuqun_zhang.txt
```

---

## Evaluation Report

| # | Question                                                   | Expected Answer                     | System Response (Summarized)                                                       | Retrieval Quality | Response Accuracy  |
| - | ---------------------------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------- | ----------------- | ------------------ |
| 1 | Which professor gives detailed feedback?                   | Professor known for useful feedback | Identified Yu Mei Huo as providing useful feedback when asked                      | Relevant          | Accurate           |
| 2 | What study method do students recommend for exams?         | Practice problems and lecture notes | Retrieved multiple references to studying lecture material and practicing problems | Relevant          | Accurate           |
| 3 | Which course is described as having the heaviest workload? | Upper-level CS course               | Retrieved reviews describing heavy workloads in advanced courses                   | Relevant          | Partially Accurate |
| 4 | What do students say about office hours for a professor?   | Helpful and informative             | Retrieved comments describing office hours as useful and responsive                | Relevant          | Accurate           |
| 5 | What internship advice appears most often?                 | Apply early and build projects      | Retrieved recommendations emphasizing projects, experience, and early applications | Relevant          | Accurate           |

**Retrieval quality:** Relevant / Partially Relevant / Off-Target

**Response accuracy:** Accurate / Partially Accurate / Inaccurate

Overall, retrieval quality was strong because most questions directly matched themes discussed in the professor review documents. Questions requiring broad comparisons across many reviews were somewhat more difficult.

---

## Failure Case Analysis

**Question that failed:**

Which professor is the easiest professor in the CSI Computer Science department?

**What the system returned:**

The system produced a partial answer based on a small number of reviews instead of identifying a clear consensus.

**Root cause (tied to a specific pipeline stage):**

The issue occurred primarily during retrieval. The documents contain subjective opinions rather than objective rankings, so the retrieved chunks did not provide enough evidence to support a definitive answer. The embedding model successfully found relevant reviews, but the available source material was insufficient.

**What I would change to fix it:**

I would collect additional reviews from more students and include a larger number of sources per professor. More review coverage would increase the likelihood that retrieval returns enough evidence for comparative questions.

---

## Spec Reflection

**One way the spec helped you during implementation:**

The planning document helped establish the architecture before coding began. By deciding on chunk size, overlap, retrieval strategy, embedding model, and evaluation questions in advance, implementation became much more organized and predictable.

**One way my implementation diverged from the spec, and why:**

The original planning document described generic course-review documents such as course reviews and internship advice. During implementation, I shifted toward professor-specific review files because they provided more focused and useful retrieval results. This change improved answer quality and made evaluation easier.

---

## Architecture

```mermaid
flowchart LR
    A[Document Ingestion] --> B[Chunking]
    B --> C[Embeddings]
    C --> D[ChromaDB Vector Store]
    D --> E[Top-k Retrieval]
    E --> F[Groq Llama 3.3 70B Generation]
```

**Libraries Used:**

* Python
* Sentence Transformers
* ChromaDB
* Groq API
* Gradio

---

## AI Usage

### Instance 1

**What I gave the AI:**

I provided my chunking strategy, retrieval requirements, and project architecture from planning.md.

**What it produced:**

It generated Python code for loading documents, splitting them into chunks, generating embeddings, and storing vectors in ChromaDB.

**What I changed or overrode:**

I adjusted file handling, integrated the code into milestone files, tested retrieval quality, and modified portions of the implementation to work correctly with my document structure.

---

### Instance 2

**What I gave the AI:**

I provided dependency errors, environment setup issues, API key problems, and debugging output from the terminal.

**What it produced:**

The AI suggested troubleshooting steps, package installation fixes, environment variable configuration corrections, and testing procedures.

**What I changed or overrode:**

I manually verified package installations, corrected environment configuration problems, updated project files, and tested the final implementation before deployment.

---

## Technologies Used

* Python 3
* Sentence Transformers
* all-MiniLM-L6-v2
* ChromaDB
* Groq API
* Llama 3.3 70B
* Gradio

---

## Author

Youssef Mohamed

College of Staten Island — Computer Science
