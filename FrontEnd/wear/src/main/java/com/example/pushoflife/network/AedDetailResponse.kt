package com.example.pushoflife.network

data class AedDetailResponse(
    val aed_address: String,
    val aed_place: String,
    val aed_location: String,
    val aed_latitude: Double,
    val aed_longitude: Double,
    val aed_number: String,
    val aed_mon_st_time: String?,
    val aed_mon_end_time: String?,
    val aed_tue_st_time: String?,
    val aed_tue_end_time: String?,
    val aed_wed_st_time: String?,
    val aed_wed_end_time: String?,
    val aed_thu_st_time: String?,
    val aed_thu_end_time: String?,
    val aed_fri_st_time: String?,
    val aed_fri_end_time: String?,
    val aed_sat_st_time: String?,
    val aed_sat_end_time: String?,
    val aed_sun_st_time: String?,
    val aed_sun_end_time: String?,
    val aed_hol_st_time: String?,
    val aed_hol_end_time: String?,
    val aed_fir_sun: String?,
    val aed_sec_sun: String?,
    val aed_thi_sun: String?,
    val aed_fou_sun: String?
)