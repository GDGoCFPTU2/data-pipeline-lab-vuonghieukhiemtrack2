from pydantic import BaseModel

# ==========================================
# ROLE 1: LEAD DATA ARCHITECT
# ==========================================

class UnifiedDocument(BaseModel):
    document_id: str
    source_type: str
    author: str
    category: str
    content: str
    timestamp: str
