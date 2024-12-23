package com.example.PushOfLife.dto.user;

import com.example.PushOfLife.entity.UserEntity;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class UserListDTO {

    @JsonProperty(value = "user_name", required = false)
    private String userName;

    @JsonProperty(value = "user_birthday", required = false)
    private String userBirthday;

    @JsonProperty(value = "user_gender", required = false)
    private String userGender;

    @JsonProperty(value = "user_phone", required = false)
    private String userPhone;

    @JsonProperty(value = "user_disease")
    private String userDisease;

    @JsonProperty(value = "user_protector", required = false)
    private String userProtector;

    public void fromEntity(UserEntity user) {
        this.userName = user.getUserName();
        this.userBirthday = user.getUserBirthday();
        this.userGender = user.getUserGender();
        this.userPhone = user.getUserPhone();
        this.userDisease = user.getUserDisease();
        this.userProtector = user.getUserProtector();
    }
}
