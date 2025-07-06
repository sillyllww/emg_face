// 获取DOM元素
const connectDeviceBtn = document.getElementById('connectDeviceBtn');
const deviceModal = document.getElementById('deviceModal');
const closeModalBtn = document.getElementById('closeModalBtn');
const scanBtn = document.getElementById('scanBtn');
const deviceList = document.getElementById('deviceList');
const deviceStatusDot = document.getElementById('deviceStatusDot');
const deviceStatusText = document.getElementById('deviceStatusText');
const startBtn = document.querySelector('.start-btn');
const pauseBtn = document.querySelector('.pause-btn');
const stopBtn = document.querySelector('.stop-btn');

// 调试面板相关变量和元素
const debugBtn = document.getElementById('debugBtn');
const paramsPanel = document.getElementById('paramsPanel');
const channelSelector = document.getElementById('channelSelector');
const channelValue = document.getElementById('channelValue');
const controlBtn = document.getElementById('controlBtn');
const dataMonitor = document.getElementById('dataMonitor');
const clearDataBtn = document.getElementById('clearDataBtn');

// 蓝牙设备相关变量
let bluetoothDevice = null;
let isScanning = false;

// GATT操作队列
let gattQueue = [];
let isGattOperationInProgress = false;

// 在全局变量区域添加参数控制变量
let paramA = 14.8; // 默认值改为毫秒
let paramB = 14.8;
let paramC = 14.8;
let paramD = 14.8;
let currentHigh = 5;
let currentLow = 5;

// 打开设备连接弹窗
connectDeviceBtn.addEventListener('click', () => {
  deviceModal.classList.add('active');
});

// 关闭设备连接弹窗
closeModalBtn.addEventListener('click', () => {
  deviceModal.classList.remove('active');
});

// 点击弹窗外部关闭弹窗
deviceModal.addEventListener('click', (e) => {
  if (e.target === deviceModal) {
    deviceModal.classList.remove('active');
  }
});

// 扫描蓝牙设备
scanBtn.addEventListener('click', async () => {
  if (isScanning) return;
  
  try {
    isScanning = true;
    scanBtn.disabled = true;
    scanBtn.textContent = '扫描中...';
    deviceList.innerHTML = '<div style="text-align: center; padding: 20px; color: #666;">正在扫描设备...</div>';

    // 请求蓝牙设备
    bluetoothDevice = await navigator.bluetooth.requestDevice({
      acceptAllDevices: true,
      optionalServices: [
        'generic_access',
        'battery_service',
        'device_information',
        '00001800-0000-1000-8000-00805f9b34fb',
        '00001801-0000-1000-8000-00805f9b34fb',
        '0000180a-0000-1000-8000-00805f9b34fb',
        '0000180f-0000-1000-8000-00805f9b34fb',
        '0000ff10-0000-1000-8000-00805f9b34fb',
        '0000ff00-0000-1000-8000-00805f9b34fb',
        '0000ffb0-0000-1000-8000-00805f9b34fb'
      ]
    });

    // 显示找到的设备
    deviceList.innerHTML = `
      <div class="device-item" data-device-id="${bluetoothDevice.id}">
        <div>
          <div class="device-name">${bluetoothDevice.name || '未知设备'}</div>
          <div class="device-id">ID: ${bluetoothDevice.id}</div>
        </div>
      </div>
    `;
    
    addDataLog("找到蓝牙设备", {
      名称: bluetoothDevice.name || '未知设备',
      ID: bluetoothDevice.id
    });

    // 点击设备进行连接
    const deviceItem = deviceList.querySelector('.device-item');
    deviceItem.addEventListener('click', async () => {
      try {
        // 连接到设备
        const server = await bluetoothDevice.gatt.connect();
        
        // 更新设备状态
        deviceStatusDot.classList.add('connected');
        deviceStatusText.textContent = '已连接设备';
        deviceStatusText.style.color = '#28A745';
        
        // 启用训练按钮和控制按钮
        startBtn.disabled = false;
        controlBtn.disabled = false;
        controlBtn.style.backgroundColor = '#28A745';
        
        // 关闭弹窗
        deviceModal.classList.remove('active');
        
        // 更新连接按钮状态
        connectDeviceBtn.textContent = '断开连接';
        connectDeviceBtn.style.backgroundColor = '#DC3545';
        
        addDataLog("设备连接成功", {
          名称: bluetoothDevice.name || '未知设备',
          ID: bluetoothDevice.id
        });
        
        // 监听设备断开连接
        bluetoothDevice.addEventListener('gattserverdisconnected', onDisconnected);
      } catch (error) {
        console.error('连接设备失败:', error);
        alert('连接设备失败，请重试');
      }
    });

  } catch (error) {
    console.error('扫描设备失败:', error);
    deviceList.innerHTML = '<div style="text-align: center; padding: 20px; color: #DC3545;">扫描设备失败，请重试</div>';
  } finally {
    isScanning = false;
    scanBtn.disabled = false;
    scanBtn.textContent = '扫描设备';
  }
});

// 设备断开连接处理
function onDisconnected() {
  // 更新设备状态
  deviceStatusDot.classList.remove('connected');
  deviceStatusText.textContent = '未连接设备';
  deviceStatusText.style.color = '#666666';
  
  // 禁用训练按钮
  startBtn.disabled = true;
  pauseBtn.disabled = true;
  stopBtn.disabled = true;
  
  // 更新连接按钮状态
  connectDeviceBtn.textContent = '连接设备';
  connectDeviceBtn.style.backgroundColor = '#007BFF';
  
  // 清除设备引用和缓存
  bluetoothDevice = null;
  deviceServiceCache = null;
  deviceCharacteristicCache = null;

  // 禁用控制按钮
  controlBtn.disabled = true;
  controlBtn.style.backgroundColor = '#6c757d';
}

// 断开设备连接
connectDeviceBtn.addEventListener('click', async () => {
  if (bluetoothDevice && bluetoothDevice.gatt.connected) {
    try {
      await bluetoothDevice.gatt.disconnect();
      // 更新按钮状态
      controlBtn.disabled = true;
      controlBtn.style.backgroundColor = '#6c757d';
    } catch (error) {
      console.error('断开连接失败:', error);
    }
  } else {
    deviceModal.classList.add('active');
  }
});

// 训练控制按钮事件处理
startBtn.addEventListener('click', () => {
  // TODO: 实现开始训练逻辑
  startBtn.disabled = true;
  pauseBtn.disabled = false;
  stopBtn.disabled = false;
});

pauseBtn.addEventListener('click', () => {
  // TODO: 实现暂停训练逻辑
  const isPaused = pauseBtn.textContent === '继续';
  pauseBtn.textContent = isPaused ? '暂停' : '继续';
});

stopBtn.addEventListener('click', () => {
  // TODO: 实现停止训练逻辑
  startBtn.disabled = false;
  pauseBtn.disabled = true;
  stopBtn.disabled = true;
  pauseBtn.textContent = '暂停';
});

// 调试面板显示/隐藏
debugBtn.addEventListener('click', () => {
  paramsPanel.classList.toggle('active');
  debugBtn.textContent = paramsPanel.classList.contains('active') ? '关闭调试' : '调试面板';
  
  // 如果面板打开，启用刺激按钮
  if (paramsPanel.classList.contains('active') && bluetoothDevice && bluetoothDevice.gatt.connected) {
    startStimBtn.disabled = false;
  }
});

// 通道选择
channelSelector.addEventListener('input', (e) => {
  channelValue.textContent = e.target.value;
});

// 按钮状态
let isButtonPressed = false;
let lastOperationTime = 0;

// 添加毫秒转换为高低字节的函数
function msToHighLowBytes(ms) {
    // 将毫秒转换为10微秒单位的值
    const value = Math.round(ms * 100); // 1ms = 100 * 10us
    return {
        highByte: (value >> 8) & 0xFF,
        lowByte: value & 0xFF
    };
}

// 添加参数更新函数
function updateParams() {
    // 从滑块获取毫秒值
    paramA = parseFloat(document.getElementById('paramA').value);
    paramB = parseFloat(document.getElementById('paramB').value);
    paramC = parseFloat(document.getElementById('paramC').value);
    paramD = parseFloat(document.getElementById('paramD').value);
    currentHigh = parseInt(document.getElementById('currentHigh').value);
    currentLow = parseInt(document.getElementById('currentLow').value);

    // 更新通道值
    const channelSelector = document.getElementById('channelSelector');
    const channelValue = document.getElementById('channelValue');
    if (channelSelector && channelValue) {
        channelValue.textContent = channelSelector.value;
    }

    // 更新显示值，保留一位小数
    document.getElementById('paramAValue').textContent = paramA.toFixed(1) + 'ms';
    document.getElementById('paramBValue').textContent = paramB.toFixed(1) + 'ms';
    document.getElementById('paramCValue').textContent = paramC.toFixed(1) + 'ms';
    document.getElementById('paramDValue').textContent = paramD.toFixed(1) + 'ms';
    document.getElementById('currentHighValue').textContent = currentHigh;
    document.getElementById('currentLowValue').textContent = currentLow;
}

// 修改按钮事件处理
controlBtn.addEventListener('mousedown', async () => {
  if (!bluetoothDevice || !bluetoothDevice.gatt.connected) {
    alert('请先连接设备');
    return;
  }
  
  // 防止连续快速操作
  const now = Date.now();
  if (now - lastOperationTime < 500) {
    addDataLog("操作过于频繁，请稍候再试");
    return;
  }
  lastOperationTime = now;
  
  // 防止重复操作
  if (isButtonPressed) return;
  isButtonPressed = true;

  // 获取当前选中的通道
  const selectedChannel = parseInt(channelValue.textContent);
  
  // 计算ABCD的高低字节，使用毫秒转换函数
  const a = msToHighLowBytes(paramA);
  const b = msToHighLowBytes(paramB);
  const c = msToHighLowBytes(paramC);
  const d = msToHighLowBytes(paramD);
  
  // 创建启动数据
  const pressCommand = [
    selectedChannel,
    1,
    a.highByte, a.lowByte,
    b.highByte, b.lowByte,
    c.highByte, c.lowByte,
    d.highByte, d.lowByte,
    currentHigh,
    currentLow
  ];

  try {
    // 改变按钮样式
    controlBtn.style.backgroundColor = "#DC3545";
    controlBtn.textContent = "松开停止";
    
    // 在控制台打印数据
    console.log('按下按钮，发送数据:', pressCommand);
    
    // 显示格式化的数据信息
    addDataLog("按下按钮，发送数据", pressCommand);

    // 通过蓝牙发送数据到设备
    await sendBluetoothData(bluetoothDevice, JSON.stringify({
      "mode": "",
      "motors": pressCommand
    }));

  } catch (error) {
    console.error('发送数据失败:', error);
    addDataLog("发送数据失败", error.message);
    
    // 恢复按钮样式
    controlBtn.style.backgroundColor = "#28A745";
    controlBtn.textContent = "按下发送数据，松开停止";
    
    isButtonPressed = false;
  }
});

controlBtn.addEventListener('mouseup', async () => {
  if (!bluetoothDevice || !bluetoothDevice.gatt.connected || !isButtonPressed) {
    return;
  }
  
  isButtonPressed = false;

  // 获取当前选中的通道
  const selectedChannel = parseInt(channelValue.textContent);
  
  // 计算ABCD的高低字节，使用毫秒转换函数
  const a = msToHighLowBytes(paramA);
  const b = msToHighLowBytes(paramB);
  const c = msToHighLowBytes(paramC);
  const d = msToHighLowBytes(paramD);
  
  // 创建停止数据
  const releaseCommand = [
    selectedChannel,
    0,
    a.highByte, a.lowByte,
    b.highByte, b.lowByte,
    c.highByte, c.lowByte,
    d.highByte, d.lowByte,
    0,
    0
  ];
  
  try {
    // 先恢复按钮样式
    controlBtn.style.backgroundColor = "#28A745";
    controlBtn.textContent = "按下发送数据，松开停止";
    
    // 在控制台打印数据
    console.log('松开按钮，发送数据:', releaseCommand);
    
    // 显示格式化的数据信息
    addDataLog("松开按钮，发送数据", releaseCommand);

    // 通过蓝牙发送数据到设备
    await sendBluetoothData(bluetoothDevice, JSON.stringify({
      "mode": "",
      "motors": releaseCommand
    }));
    
  } catch (error) {
    console.error('发送停止数据失败:', error);
    addDataLog("发送停止数据失败", error.message);
  }
});

// 添加触摸支持
controlBtn.addEventListener('touchstart', async (e) => {
  e.preventDefault(); // 防止触发鼠标事件
  
  if (!bluetoothDevice || !bluetoothDevice.gatt.connected) {
    alert('请先连接设备');
    return;
  }
  
  // 防止连续快速操作
  const now = Date.now();
  if (now - lastOperationTime < 500) {
    addDataLog("操作过于频繁，请稍候再试");
    return;
  }
  lastOperationTime = now;
  
  // 防止重复操作
  if (isButtonPressed) return;
  isButtonPressed = true;

  // 获取当前选中的通道
  const selectedChannel = parseInt(channelValue.textContent);
  
  // 计算ABCD的高低字节，使用毫秒转换函数
  const a = msToHighLowBytes(paramA);
  const b = msToHighLowBytes(paramB);
  const c = msToHighLowBytes(paramC);
  const d = msToHighLowBytes(paramD);
  const highCurrent = currentHigh;
  const lowCurrent = currentLow;
  
  // 创建启动数据
  const pressCommand = [
    selectedChannel,
    1,
    a.highByte, a.lowByte,
    b.highByte, b.lowByte,
    c.highByte, c.lowByte,
    d.highByte, d.lowByte,
    highCurrent,
    lowCurrent
  ];

  try {
    // 改变按钮样式
    controlBtn.style.backgroundColor = "#DC3545";
    controlBtn.textContent = "松开停止";
    
    // 在控制台打印数据
    console.log('触摸开始，发送数据:', pressCommand);
    
    // 显示格式化的数据信息
    addDataLog("触摸开始，发送数据", pressCommand);

    // 通过蓝牙发送数据到设备
    await sendBluetoothData(bluetoothDevice, JSON.stringify({
      "mode": "",
      "motors": pressCommand
    }));

  } catch (error) {
    console.error('发送数据失败:', error);
    addDataLog("发送数据失败", error.message);
    
    // 恢复按钮样式
    controlBtn.style.backgroundColor = "#28A745";
    controlBtn.textContent = "按下发送数据，松开停止";
    
    isButtonPressed = false;
  }
});

controlBtn.addEventListener('touchend', async (e) => {
  e.preventDefault(); // 防止触发鼠标事件
  
  if (!bluetoothDevice || !bluetoothDevice.gatt.connected || !isButtonPressed) {
    return;
  }
  
  // 防止连续快速操作
  const now = Date.now();
  if (now - lastOperationTime < 500) {
    addDataLog("操作过于频繁，请稍候再试");
    return;
  }
  lastOperationTime = now;
  
  isButtonPressed = false;

  // 获取当前选中的通道
  const selectedChannel = parseInt(channelValue.textContent);
  
  // 计算ABCD的高低字节，使用毫秒转换函数
  const a = msToHighLowBytes(paramA);
  const b = msToHighLowBytes(paramB);
  const c = msToHighLowBytes(paramC);
  const d = msToHighLowBytes(paramD);
  
  // 创建停止数据
  const releaseCommand = [
    selectedChannel,
    0,
    a.highByte, a.lowByte,
    b.highByte, b.lowByte,
    c.highByte, c.lowByte,
    d.highByte, d.lowByte,
    0,
    0
  ];
  
  try {
    // 先恢复按钮样式
    controlBtn.style.backgroundColor = "#28A745";
    controlBtn.textContent = "按下发送数据，松开停止";
    
    // 在控制台打印数据
    console.log('触摸结束，发送数据:', releaseCommand);
    
    // 显示格式化的数据信息
    addDataLog("触摸结束，发送数据", releaseCommand);

    // 通过蓝牙发送数据到设备
    await sendBluetoothData(bluetoothDevice, JSON.stringify({
      "mode": "",
      "motors": releaseCommand
    }));
    
  } catch (error) {
    console.error('发送停止数据失败:', error);
    addDataLog("发送停止数据失败", error.message);
  }
});

// 蓝牙设备的服务和特性缓存
let deviceServiceCache = null;
let deviceCharacteristicCache = null;

// 添加日志到数据监控区域
function addDataLog(message, data = null) {
  const now = new Date();
  const timestamp = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}.${now.getMilliseconds().toString().padStart(3, '0')}`;
  
  let logMessage = `[${timestamp}] ${message}`;
  
  if (data) {
    if (typeof data === 'object') {
      logMessage += `<br><pre style="margin: 4px 0; padding: 6px; background-color: #e9ecef; border-radius: 4px; overflow-x: auto;">${JSON.stringify(data, null, 2)}</pre>`;
    } else {
      logMessage += `<br>${data}`;
    }
  }
  
  dataMonitor.innerHTML += `<div>${logMessage}</div>`;
  dataMonitor.scrollTop = dataMonitor.scrollHeight;
}

// 清除数据监控区域
clearDataBtn.addEventListener('click', () => {
  dataMonitor.innerHTML = '';
});

// 特性选择模态框
let characteristicSelectionModal = null;
let availableCharacteristics = [];

// 创建特性选择模态框
function createCharacteristicSelectionModal() {
  // 如果已经存在则返回
  if (document.getElementById('characteristicModal')) {
    return document.getElementById('characteristicModal');
  }
  
  // 创建模态框元素
  const modal = document.createElement('div');
  modal.id = 'characteristicModal';
  modal.style.position = 'fixed';
  modal.style.top = '0';
  modal.style.left = '0';
  modal.style.width = '100%';
  modal.style.height = '100%';
  modal.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
  modal.style.display = 'none';
  modal.style.justifyContent = 'center';
  modal.style.alignItems = 'center';
  modal.style.zIndex = '2000';
  
  // 创建模态框内容
  const content = document.createElement('div');
  content.style.backgroundColor = '#FFFFFF';
  content.style.borderRadius = '12px';
  content.style.padding = '24px';
  content.style.width = '500px';
  content.style.maxWidth = '90%';
  content.style.maxHeight = '80%';
  content.style.overflowY = 'auto';
  
  // 创建标题
  const header = document.createElement('div');
  header.style.display = 'flex';
  header.style.justifyContent = 'space-between';
  header.style.alignItems = 'center';
  header.style.marginBottom = '20px';
  
  const title = document.createElement('h2');
  title.textContent = '选择蓝牙特性';
  title.style.fontSize = '18px';
  title.style.fontWeight = '600';
  title.style.margin = '0';
  
  const closeBtn = document.createElement('button');
  closeBtn.innerHTML = '&times;';
  closeBtn.style.background = 'none';
  closeBtn.style.border = 'none';
  closeBtn.style.fontSize = '24px';
  closeBtn.style.cursor = 'pointer';
  closeBtn.style.padding = '0';
  closeBtn.onclick = () => {
    modal.style.display = 'none';
  };
  
  header.appendChild(title);
  header.appendChild(closeBtn);
  
  // 创建特性列表容器
  const listContainer = document.createElement('div');
  listContainer.id = 'characteristicsList';
  listContainer.style.marginBottom = '20px';
  
  content.appendChild(header);
  content.appendChild(listContainer);
  
  modal.appendChild(content);
  document.body.appendChild(modal);
  
  return modal;
}

// 选择特性弹窗
function showCharacteristicSelectionModal(characteristics) {
  return new Promise((resolve) => {
    const modal = createCharacteristicSelectionModal();
    const listContainer = document.getElementById('characteristicsList');
    
    // 清空旧内容
    listContainer.innerHTML = '';
    
    // 添加说明
    const description = document.createElement('p');
    description.textContent = '以下是可用的蓝牙特性。请选择一个特性用于数据通信：';
    description.style.marginBottom = '16px';
    listContainer.appendChild(description);
    
    // 添加警告
    const warning = document.createElement('p');
    warning.textContent = '注意：即使特性没有显示为可写，也可以尝试选择它，有些设备可能支持写操作但未正确声明。';
    warning.style.color = '#DC3545';
    warning.style.marginBottom = '16px';
    listContainer.appendChild(warning);
    
    // 添加特性列表
    characteristics.forEach((characteristic, index) => {
      const props = [];
      if (characteristic.properties.write) props.push("write");
      if (characteristic.properties.writeWithoutResponse) props.push("writeWithoutResponse");
      if (characteristic.properties.read) props.push("read");
      if (characteristic.properties.notify) props.push("notify");
      
      const item = document.createElement('div');
      item.style.padding = '12px';
      item.style.border = '1px solid #E0E0E0';
      item.style.borderRadius = '8px';
      item.style.marginBottom = '8px';
      item.style.cursor = 'pointer';
      
      item.innerHTML = `
        <div style="font-weight:500;">特性 ${index + 1}</div>
        <div style="font-size:14px;">UUID: ${characteristic.uuid}</div>
        <div style="font-size:12px;color:#666;">属性: ${props.join(", ") || "无"}</div>
      `;
      
      // 根据属性添加不同的背景色
      if (characteristic.properties.write || characteristic.properties.writeWithoutResponse) {
        item.style.backgroundColor = 'rgba(40, 167, 69, 0.1)';
      }
      
      item.onclick = () => {
        resolve(characteristic);
        modal.style.display = 'none';
      };
      
      listContainer.appendChild(item);
    });
    
    // 显示模态框
    modal.style.display = 'flex';
  });
}

// 获取可写特性
async function getWritableCharacteristic(device) {
  if (deviceCharacteristicCache) {
    return deviceCharacteristicCache;
  }
  
  try {
    // 目标特征UUID
    const targetCharacteristicUuid = "0000ff11-0000-1000-8000-00805f9b34fb";
      
      addDataLog("连接蓝牙设备GATT服务器...");
      
      // 连接到设备的GATT服务器
      const server = await device.gatt.connect();
      
      // 获取所有服务
      const services = await server.getPrimaryServices();
      addDataLog(`找到 ${services.length} 个服务`);
      
      // 存储所有特性
      availableCharacteristics = [];
    
    // 首先尝试直接通过UUID获取特定特征
    try {
      // 遍历每个服务
      for (const service of services) {
        addDataLog(`检查服务: ${service.uuid}`);
        
        try {
          // 尝试获取目标特征
          const characteristic = await service.getCharacteristic(targetCharacteristicUuid);
          addDataLog(`在服务 ${service.uuid} 中找到目标特征 ${targetCharacteristicUuid}`, "使用该特性进行通信");
          deviceServiceCache = service;
          deviceCharacteristicCache = characteristic;
          return characteristic;
        } catch (e) {
          // 这个服务中没有目标特征，继续检查下一个
          addDataLog(`服务 ${service.uuid} 中未找到目标特征，继续查找其他服务...`);
        }
      }
    } catch (specificError) {
      addDataLog(`直接获取目标特征失败: ${specificError.message}`, "将尝试获取所有特征");
    }
    
    // 如果直接获取失败，遍历所有特征
    for (const service of services) {
      // 获取服务的所有特性
      const characteristics = await service.getCharacteristics();
      addDataLog(`服务 ${service.uuid} 有 ${characteristics.length} 个特性`);
      
      // 收集所有特性
      characteristics.forEach(characteristic => {
        availableCharacteristics.push(characteristic);
        
        const props = [];
        if (characteristic.properties.write) props.push("write");
        if (characteristic.properties.writeWithoutResponse) props.push("writeWithoutResponse");
        if (characteristic.properties.read) props.push("read");
        if (characteristic.properties.notify) props.push("notify");
        
        addDataLog(`特性: ${characteristic.uuid}, 属性: ${props.join(", ")}`);
        
        // 检查是否为目标特征
        if (characteristic.uuid === targetCharacteristicUuid) {
          addDataLog(`找到目标特征: ${characteristic.uuid}`, "使用该特性进行通信");
          deviceServiceCache = service;
          deviceCharacteristicCache = characteristic;
        }
        // 如果还没有找到特征且这是一个可写特征
        else if (!deviceCharacteristicCache && (characteristic.properties.write || characteristic.properties.writeWithoutResponse)) {
          addDataLog(`找到可写特性: ${characteristic.uuid}`, "暂存该特性");
          deviceServiceCache = service;
          deviceCharacteristicCache = characteristic;
        }
      });
      
      // 如果在这个服务中找到了目标特征，直接返回
      if (deviceCharacteristicCache && deviceCharacteristicCache.uuid === targetCharacteristicUuid) {
        return deviceCharacteristicCache;
      }
    }
    
    // 如果找到了可写特性（虽然不是目标特性），也返回
    if (deviceCharacteristicCache) {
      addDataLog(`未找到目标特征，使用可写特征: ${deviceCharacteristicCache.uuid}`, "尝试使用该特性进行通信");
      return deviceCharacteristicCache;
    }
    
    // 如果没有找到可写特性，显示特性选择模态框
    addDataLog("未找到目标特征或可写特性，请手动选择一个特性用于通信");
    
    // 如果有任何特性可供选择
    if (availableCharacteristics.length > 0) {
      // 显示特性选择对话框
      const selectedCharacteristic = await showCharacteristicSelectionModal(availableCharacteristics);
      
      addDataLog(`已选择特性: ${selectedCharacteristic.uuid}`, "将使用该特性进行通信");
      deviceCharacteristicCache = selectedCharacteristic;
      return selectedCharacteristic;
    }
    
    throw new Error('设备未提供任何可用的特性');
  } catch (error) {
    console.error('获取可写特性失败:', error);
    addDataLog("获取可写特性失败", error.message);
    throw error;
  }
}

// 添加延迟函数
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// 蓝牙数据发送函数
async function sendBluetoothData(device, data) {
  try {
    // 等待锁定解除
    while (isGattOperationInProgress) {
      await sleep(100);
    }
    
    // 设置锁定标志
    isGattOperationInProgress = true;
    
    // 获取可写特性
    const characteristic = await getWritableCharacteristic(device);
    
    // 将数据转换为ArrayBuffer
    const encoder = new TextEncoder();
    const dataBuffer = encoder.encode(data);
    
    // 记录发送的数据到日志区域
    addDataLog("发送数据", JSON.parse(data));
    
    // 在控制台打印详细数据
    console.log('发送数据详情:', {
      时间: new Date().toISOString(),
      特性UUID: characteristic.uuid,
      数据: JSON.parse(data)
    });
    
    // 使用writeValue发送数据
    await characteristic.writeValue(dataBuffer);
    
    // 记录成功信息
    addDataLog("数据发送成功", "使用writeValue方法");
    console.log('数据发送成功 - 使用writeValue方法');
    
    // 解除锁定
    isGattOperationInProgress = false;
    
    return true;
  } catch (error) {
    console.error('蓝牙数据发送失败:', error);
    addDataLog("发送失败", error.message);
    
    // 发生错误时也解除锁定
    isGattOperationInProgress = false;
    
    throw error;
  }
} 