import java.io.BufferedReader
import java.io.FileReader
import kotlin.concurrent.fixedRateTimer
import java.util.LinkedList

fun main() {
    // PushSensor 인스턴스 생성
    val pushSensor = PushSensor()
    
    // CSV 파일 경로 (경로를 파일의 실제 위치로 설정하세요)
    val csvFilePath = "강사님_csv.csv"
    
    // 가속도 데이터를 저장할 큐
    val accelerationData = LinkedList<Float>()

    // CSV 파일 읽기
    BufferedReader(FileReader(csvFilePath)).use { reader ->
        reader.readLine() // 헤더를 무시
        reader.lineSequence().forEach { line ->
            val data = line.split(",")  // CSV에서 구분자가 콤마(,)로 되어 있는 경우
            val acceleration = data[0].toFloat()  // 첫 번째 열에 가속도 값이 있다고 가정
            accelerationData.add(acceleration)
        }
    }

    // 실시간 데이터를 시뮬레이션하기 위해 0.01초 간격으로 가속도 데이터를 전달
    fixedRateTimer("CSV-Simulation", initialDelay = 0, period = 10) {
        if (accelerationData.isNotEmpty()) {
            val currentAcceleration = accelerationData.poll()
            pushSensor.calculateCPRFeedback(currentAcceleration)
        } else {
            println("CSV 데이터가 모두 소모되었습니다.")
            this.cancel() // 모든 데이터를 사용하면 타이머 종료
        }
    }
}
