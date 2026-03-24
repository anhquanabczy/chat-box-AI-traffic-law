# data_loader.py
import streamlit as st  # <-- BẮT BUỘC PHẢI THÊM ĐỂ IN LỖI LÊN WEB
import config 
from vector_db import SimpleVectorDatabase
from retriever import Retriever
from utils import embed_legal_chunks

def load_or_create_rag_components(embedding_model_object, rag_data_prefix: str):
    vector_db_instance = SimpleVectorDatabase()
    hybrid_retriever_instance = None

    if vector_db_instance.load(rag_data_prefix):
        hybrid_retriever_instance = Retriever(vector_db_instance, bm25_save_path=f"{rag_data_prefix}_bm25.pkl")
    else:
        json_files = [
            config.JSON_FILE_PATTERN.format(i=i)
            for i in range(1, config.NUM_FILES + 1)
            if i not in config.NUMBERS_TO_SKIP
        ]

        if not json_files: 
            # Bắt lỗi 1: Không có file JSON
            st.error("🚨 THÔNG BÁO TỪ QUÂN: Danh sách json_files bị rỗng. Không tìm thấy data!")
            return None, None

        valid_chunks, embeddings_array = embed_legal_chunks(json_files, embedding_model_object)
        
        if valid_chunks and embeddings_array is not None:
            vector_db_instance.add_documents(valid_chunks, embeddings_array)
            vector_db_instance.save(rag_data_prefix)
            hybrid_retriever_instance = Retriever(vector_db_instance, bm25_save_path=f"{rag_data_prefix}_bm25.pkl")
        else: 
            # Bắt lỗi 2: embed_legal_chunks thất bại ngầm
            st.error(f"🚨 THÔNG BÁO TỪ QUÂN: Hàm embed_legal_chunks đã tịt ngòi! Số valid_chunks lấy được: {len(valid_chunks) if valid_chunks else 0}. Có phải embeddings_array bị None không? {embeddings_array is None}")
            return None, None

    return vector_db_instance, hybrid_retriever_instance
