package api

import models.*
import retrofit2.Response
import retrofit2.http.*

interface ApiService {
    @GET("legal_entities")
    suspend fun getLegalEntities(): Response<List<LegalEntity>>

    @GET("legal_entities/{id}")
    suspend fun getLegalEntity(@Path("id") id: String): Response<LegalEntity>

    @POST("legal_entities")
    suspend fun addLegalEntity(@Body entity: LegalEntity): Response<Map<String, String>>

    @PUT("legal_entities/{le_id}")
    suspend fun updateLegalEntity(
        @Path("le_id") leId: String,
        @Body entity: LegalEntity
    ): Response<Map<String, String>>

    @DELETE("legal_entities/{le_id}")
    suspend fun deleteLegalEntity(@Path("le_id") leId: String): Response<Map<String, String>>

    // Management Entities
    @GET("management_entities")
    suspend fun getManagementEntities(): Response<List<ManagementEntity>>

    @GET("management_entities/{id}")
    suspend fun getManagementEntity(@Path("id") id: String): Response<ManagementEntity>

    @POST("management_entities")
    suspend fun addManagementEntity(@Body entity: ManagementEntity): Response<Map<String, String>>

    @PUT("management_entities/{mgmt_id}")
    suspend fun updateManagementEntity(
        @Path("mgmt_id") mgmtId: String,
        @Body entity: ManagementEntity
    ): Response<Map<String, String>>

    @DELETE("management_entities/{mgmt_id}")
    suspend fun deleteManagementEntity(@Path("mgmt_id") mgmtId: String): Response<Map<String, String>>

    // Funds
    @GET("funds")
    suspend fun getFunds(): Response<List<FundMaster>>

    @GET("funds/{id}")
    suspend fun getFund(@Path("id") id: String): Response<FundMaster>

    @POST("funds")
    suspend fun addFund(@Body fund: FundMaster): Response<Map<String, String>>

    @PUT("funds/{fund_id}")
    suspend fun updateFund(
        @Path("fund_id") fundId: String,
        @Body fund: FundMaster
    ): Response<Map<String, String>>

    @DELETE("funds/{fund_id}")
    suspend fun deleteFund(@Path("fund_id") fundId: String): Response<Map<String, String>>

    // Sub Funds
    @GET("sub_funds")
    suspend fun getSubFunds(): Response<List<SubFund>>

    @GET("sub_funds/{id}")
    suspend fun getSubFund(@Path("id") id: String): Response<SubFund>

    @POST("sub_funds")
    suspend fun addSubFund(@Body subFund: SubFund): Response<Map<String, String>>

    @PUT("sub_funds/{subfund_id}")
    suspend fun updateSubFund(
        @Path("subfund_id") subfundId: String,
        @Body subFund: SubFund
    ): Response<Map<String, String>>

    @DELETE("sub_funds/{subfund_id}")
    suspend fun deleteSubFund(@Path("subfund_id") subfundId: String): Response<Map<String, String>>

    // Share Classes
    @GET("share_classes")
    suspend fun getShareClasses(): Response<List<ShareClass>>

    @GET("share_classes/{id}")
    suspend fun getShareClass(@Path("id") id: String): Response<ShareClass>

    @POST("share_classes")
    suspend fun addShareClass(@Body shareClass: ShareClass): Response<Map<String, String>>

    @PUT("share_classes/{sc_id}")
    suspend fun updateShareClass(
        @Path("sc_id") scId: String,
        @Body shareClass: ShareClass
    ): Response<Map<String, String>>

    @DELETE("share_classes/{sc_id}")
    suspend fun deleteShareClass(@Path("sc_id") scId: String): Response<Map<String, String>>

    // Health Check
    @GET("health")
    suspend fun healthCheck(): Response<Map<String, String>>
}