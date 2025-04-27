package com.sjtu.KGStudy.abnormalDataRec.utils;

import java.io.IOException;
import java.nio.file.*;
import java.util.concurrent.TimeUnit;
import org.json.JSONObject;
import java.util.Map;


public class PauseJavaUtils {
    public static void waitForDataReconstruction(String targetFloder, String targetFileName) throws IOException, InterruptedException {
        // 将传入的文件夹路径转换为Path对象
        Path path = Paths.get(targetFloder);

        // 检查目标文件夹是否存在
        if (!Files.isDirectory(path)) {
            throw new IllegalArgumentException("目标路径不是有效的文件夹: " + targetFloder);
        }

        // 创建WatchService
        try (WatchService watchService = FileSystems.getDefault().newWatchService()) {
            // 注册文件夹到WatchService，监听创建事件
            path.register(watchService, StandardWatchEventKinds.ENTRY_CREATE);

            System.out.println("监控已启动，等待文件 " + targetFileName + " 出现...");

            while (true) {
                // 阻塞直到有事件发生
                WatchKey key = watchService.poll(1, TimeUnit.SECONDS);

                if (key != null) {
                    for (WatchEvent<?> event : key.pollEvents()) {
                        // 获取事件类型和文件名
                        WatchEvent.Kind<?> kind = event.kind();
                        Path fileName = (Path) event.context();

                        // 检查是否是目标文件
                        if (kind == StandardWatchEventKinds.ENTRY_CREATE && targetFileName.equals(fileName.toString())) {
                            System.out.println("检测到目标文件: " + targetFileName);
                            Thread.sleep(500); // 模拟 0.5 秒延迟
                            /*try {
                                // 读取 JSON 文件内容为字符串
                                String content = new String(Files.readAllBytes(Paths.get(targetFloder + "/" + targetFileName)));

                                // 使用 JSONObject 解析 JSON 字符串
                                JSONObject jsonObject = new JSONObject(content);

                                // 将 JSONObject 转为 Map
                                Map<String, Object> data = jsonObject.toMap();

                                // 打印数据到前台（控制台）
                                System.out.println("JSON 数据内容：");
                                data.forEach((key_json, value_json) -> System.out.println(key_json + ": " + value_json));
                            } catch (IOException e) {
                                e.printStackTrace();
                            }*/
                            return; // 文件检测成功，退出方法
                        }
                    }
                    // 重置Key，确保可以继续接收事件
                    if (!key.reset()) {  // 如果文件夹不再可用（例如被删除、权限变更或文件系统挂载异常），reset()会返回false
                        System.out.println("监控文件夹不可用，退出...");
                        break;
                    }
                }
            }
        }
    }
}
