from fastapi import APIRouter, Depends, HTTPException
from app.db.connections import get_db_connections as get_db
from app.services.report_service import ReportService
# Keep for existing generate_report if used elsewhere
from app.schemas.report import ReportData
# Assuming you have this for user auth
# from app.api.dependencies import get_current_active_user # Ensure this is defined in dependencies.py if used
from app.models.user import User  # Assuming you have this
from fastapi.responses import StreamingResponse
import io
# Needed for type hinting if ReportService expects it
from app.llm.rag_system import RAGSystem
from app.api.dependencies import get_rag_system_for_profile  # Updated import
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/report", tags=["report"])


# Dependency to get ReportService instance
def get_report_service(
    db_clients=Depends(get_db),
    rag_system_instance: RAGSystem = Depends(
        get_rag_system_for_profile)  # Use common dependency
):
    _, mongodb_client = db_clients
    return ReportService(mongo_db=mongodb_client, rag_system=rag_system_instance)


@router.get("/generate/{user_id}", response_class=StreamingResponse)
async def generate_user_report_pdf(
    user_id: str,
    report_service: ReportService = Depends(get_report_service)
):
    try:
        report_string = await report_service.generate_llm_filled_template_report_string(user_id)
        print(report_string)
        return HTMLResponse(content=report_string)
        # pdf_bytes = await report_service.html_to_pdf_with_playwright(report_string) 

        # return StreamingResponse(
        #     io.BytesIO(pdf_bytes),
        #     media_type="application/pdf",
        #     headers={
        #         "Content-Disposition": f"attachment; filename=career_report_{user_id}.pdf"
        #     }
        # )
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )
