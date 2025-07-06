// ========== 全局错误处理 ==========
window.addEventListener('error', (event) => {
  console.error('全局JavaScript错误:', event.error);
  console.error('错误位置:', event.filename, '行号:', event.lineno);
  event.preventDefault(); // 阻止默认的错误处理（防止页面刷新）
  return true;
});

window.addEventListener('unhandledrejection', (event) => {
  console.error('未处理的Promise拒绝:', event.reason);
  event.preventDefault(); // 阻止默认的错误处理
});

// ========== 九个动作示例信息 ==========
const actions = [
  { name: '动作一：抬眉', img: 'images/action1.png', video: 'videos/action1.mp4' },
  { name: '动作二：皱眉', img: 'images/action2.png', video: 'videos/action2.mp4' },
  { name: '动作三：闭眼', img: 'images/action3.png', video: 'videos/action3.mp4' },
  { name: '动作四：鼓腮', img: 'images/action4.png', video: 'videos/action4.mp4' },
  { name: '动作五：撅嘴', img: 'images/action5.png', video: 'videos/action5.mp4' },
  { name: '动作六：微笑', img: 'images/action6.png', video: 'videos/action6.mp4' },
  { name: '动作七：咧嘴笑', img: 'images/action7.jpg', video: 'videos/action7.mp4' },
  { name: '动作八：龇牙', img: 'images/action8.jpg', video: 'videos/action8.mp4' },
  { name: '动作九：耸鼻', img: 'images/action9.png', video: 'videos/action9.mp4' },
  { name: '动作十：静息', img: 'images/action10.png', video: '' }
];
let currentActionIndex = 0;
let isSequencePlaying = false;
let isPaused = false;
let currentPhase = '';       // 'intro', 'showRef', 'video', 'hold', 'countdown', 'relax'
let performTimeRemaining = 0;
let restTimeRemaining = 0;

// 页面元素引用
let countdownNumberElem, centerOverlay, overlayText, bottomOverlay, bottomText, actionImage, previewVideo;
let startResetBtn, pauseContinueBtn, taskTitleElem, barFill, progressLabel;

window.addEventListener('load', () => {
  // ==== 获取 DOM 元素 ====
  countdownNumberElem = document.getElementById('countdownNumber');
  centerOverlay = document.getElementById('centerOverlay');
  overlayText = document.getElementById('overlayText');
  bottomOverlay = document.getElementById('bottomOverlay');
  bottomText = document.getElementById('bottomText');
  actionImage = document.getElementById('actionImage');
  previewVideo = document.getElementById('previewVideo');
  startResetBtn = document.getElementById('startResetBtn');
  pauseContinueBtn = document.getElementById('pauseContinueBtn');
  taskTitleElem = document.getElementById('taskTitle');
  barFill = document.getElementById('barFill');
  progressLabel = document.getElementById('progressLabel');

  // ==== 按钮初始状态 ====
  pauseContinueBtn.disabled = true;

  // ==== 绑定事件 ====
  startResetBtn.addEventListener('click', () => {
    if (!isSequencePlaying) {
      startSequence();
    } else {
      resetSequence();
    }
  });
  pauseContinueBtn.addEventListener('click', () => {
    if (!isPaused) {
      pauseSequence();
    } else {
      resumeSequence();
    }
  });

  // ==== 记录按钮 ====
  const exportBtnEl = document.getElementById('exportBtn');
  exportBtnEl.addEventListener('click', () => {
    console.log('EMG数据记录由Python服务器自动处理');
    console.log('训练开始时会自动记录，训练结束时会自动导出');
  });
});

// ========== 序列控制 ==========
async function startSequence() {
  console.log('startSequence 触发');
  isSequencePlaying = true;
  isPaused = false;
  startResetBtn.innerText = '重置';
  pauseContinueBtn.disabled = false;
  pauseContinueBtn.innerText = '暂停';
  speechSynthesis.cancel();

  for (currentActionIndex = 0; currentActionIndex < actions.length; currentActionIndex++) {
    if (!isSequencePlaying) break;
    const action = actions[currentActionIndex];
    console.log(`开始第 ${currentActionIndex + 1} 个动作：`, action.name);

    // 每个动作开始时更新进度
    progressLabel.innerText = `${currentActionIndex + 1}/${actions.length}`;

    // 1. 居中灰框 + 文本 + 语音 "动作X：名称"
    currentPhase = 'intro';
    previewVideo.style.display = 'none';
    actionImage.style.display = 'none';
    countdownNumberElem.style.visibility = 'hidden';
    bottomOverlay.style.visibility = 'hidden';
    centerOverlay.style.visibility = 'visible';
    overlayText.innerText = action.name;
    taskTitleElem.innerText = action.name;
    await speak(action.name);

    // 2. 隐藏灰框，展示示例图片 + 语音 "动作参考如下" 3 秒
    centerOverlay.style.visibility = 'hidden';
    currentPhase = 'showRef';
    actionImage.src = action.img;
    actionImage.style.display = 'block';
    taskTitleElem.innerText = action.name;
    await speak('动作参考如下');
    await waitSeconds(3);

    // 3. "叮"提示音 + 播放视频两遍
    actionImage.style.display = 'none';
    currentPhase = 'video';
    await playDing();
    previewVideo.src = action.video;
    previewVideo.style.display = 'block';
    for (let i = 0; i < 2; i++) {
      if (!isSequencePlaying) break;
      console.log(`播放第 ${i+1} 次视频`);
      previewVideo.currentTime = 0;
      previewVideo.play();
      await waitForVideoEnd();
      console.log(`第 ${i+1} 次视频结束`);
    }
    if (!isSequencePlaying) break;

    // 4. 展示示例图片 + 底部灰框提示 "请保持以下动作5秒" + 语音
    currentPhase = 'hold';
    previewVideo.style.display = 'none';
    actionImage.style.display = 'block';
    bottomOverlay.style.visibility = 'visible';
    bottomText.innerText = '请保持以下动作5秒';
    await speak('在听到提示音后，请保持该动作5秒');

    // 5. "叮"提示音 + 隐藏底部灰框 + 右上角倒计时 5 秒 + 语音倒计时同时进行
    await playDing();
    bottomOverlay.style.visibility = 'hidden';
    currentPhase = 'countdown';
    countdownNumberElem.style.visibility = 'visible';
    
    // 获取用户信息并发送开始采集EMG数据指令
    const userName = document.querySelector('input[name="name"]')?.value || 'unknown';
    const userAge = document.querySelector('input[name="age"]')?.value || 'NA';
    
    try {
      await sendCommandToPython('start_recording', {
        action_label: currentActionIndex,
        user_name: userName,
        user_age: userAge
      });
      console.log(`🔴 发送开始记录指令 - 动作${currentActionIndex}`);
      
    } catch (error) {
      console.error('发送开始记录指令失败:', error);
    }
    
    // 立即开始语音播报倒计时，不等待
    speak('5，4，3，2，1').catch(console.error);
    
    // 倒计时数字显示
    for (let i = 5; i >= 1; i--) {
      if (!isSequencePlaying) break;
      if (isPaused) { await waitUntilResumed(); }
      countdownNumberElem.innerText = i;
      await waitSeconds(1);
    }
    
    // 倒计时完成后播放完成语音
    speak('恭喜完成，放松一会吧。').catch(console.error);
    countdownNumberElem.style.visibility = 'hidden';
    if (!isSequencePlaying) break;

    // 6. 放松 10 秒；直接展示下一个任务的图片，同时显示提示
    currentPhase = 'relax';
    restTimeRemaining = 10;
    
    // 如果有下一个动作，直接显示下一个动作的图片和提示
    if (currentActionIndex < actions.length - 1) {
      const nextAction = actions[currentActionIndex + 1];
      // 显示下一个动作的图片
      actionImage.src = nextAction.img;
      actionImage.style.display = 'block';
      
      // 显示提示文字
      const nextText = `即将进行下一动作。${nextAction.name}`;
      centerOverlay.style.visibility = 'visible';
      overlayText.innerText = nextText;
      taskTitleElem.innerText = nextText;
      
      // 播放语音提示
      speak(nextText).catch(console.error);
    } else {
      // 如果是最后一个动作，隐藏图片
      actionImage.style.display = 'none';
    }
    
    // 放松倒计时 10 秒
    for (let t = 10; t >= 1; t--) {
      if (!isSequencePlaying) break;
      if (isPaused) { await waitUntilResumed(); }
      
      updateProgressBar((10 - t) / 10);
      progressLabel.innerText = `放松：${t}秒`;
      await waitSeconds(1);
    }
    centerOverlay.style.visibility = 'hidden';
    updateProgressBar(0);
    if (!isSequencePlaying) break;
  }

  // 整个序列结束
  isSequencePlaying = false;
  taskTitleElem.innerText = '任务完成';
  previewVideo.style.display = 'none';
  actionImage.style.display = 'none';
  centerOverlay.style.visibility = 'hidden';
  bottomOverlay.style.visibility = 'hidden';
  countdownNumberElem.style.visibility = 'hidden';
  startResetBtn.innerText = '开始';
  pauseContinueBtn.disabled = true;
  progressLabel.innerText = '1/9';
  updateProgressBar(0);
  console.log('序列结束');
  
  // 所有任务完成后保存数据
  try {
    await sendCommandToPython('save_all_data', {});
    console.log('🎉 已发送保存所有数据的指令');
  } catch (error) {
    console.error('发送保存指令失败:', error);
  }
}

function pauseSequence() {
  if (!isSequencePlaying) return;
  isPaused = true;
  pauseContinueBtn.innerText = '继续';
  if (currentPhase === 'video') {
    previewVideo.pause();
  }
  speechSynthesis.pause();
}

function resumeSequence() {
  if (!isSequencePlaying) return;
  isPaused = false;
  pauseContinueBtn.innerText = '暂停';
  if (currentPhase === 'video') {
    previewVideo.play();
  }
  speechSynthesis.resume();
}

function resetSequence() {
  isSequencePlaying = false;
  isPaused = false;
  currentPhase = '';
  
  previewVideo.pause();
  previewVideo.style.display = 'none';
  actionImage.style.display = 'none';
  centerOverlay.style.visibility = 'hidden';
  bottomOverlay.style.visibility = 'hidden';
  countdownNumberElem.style.visibility = 'hidden';
  speechSynthesis.cancel();
  taskTitleElem.innerText = '任务暂停';
  updateProgressBar(0);
  progressLabel.innerText = '1/9';
  startResetBtn.innerText = '开始';
  pauseContinueBtn.disabled = true;
  pauseContinueBtn.innerText = '暂停';
  console.log('已重置序列');
}

function waitSeconds(sec) {
  return new Promise(resolve => {
    let remaining = sec;
    const interval = setInterval(() => {
      if (!isPaused) {
        remaining--;
      }
      if (remaining <= 0) {
        clearInterval(interval);
        resolve();
      }
    }, 1000);
  });
}

function waitUntilResumed() {
  return new Promise(resolve => {
    const check = setInterval(() => {
      if (!isPaused) {
        clearInterval(check);
        resolve();
      }
    }, 200);
  });
}

function updateProgressBar(fraction) {
  barFill.style.width = `${fraction * 100}%`;
}

// ========== SpeechSynthesis 辅助函数 ==========
function speak(text) {
  return new Promise((resolve) => {
    try {
      console.log(`🎤 开始语音播报: "${text}"`);
      
      // 检查语音合成是否可用
      if (!('speechSynthesis' in window)) {
        console.warn('浏览器不支持语音合成，跳过语音播报');
        resolve();
        return;
      }

      const utterance = new SpeechSynthesisUtterance(text);
      
      // 设置较短的超时机制（5秒超时）
      const timeout = setTimeout(() => {
        console.warn(`语音播报超时: "${text}"`);
        try {
          speechSynthesis.cancel();
        } catch (e) {
          console.warn('取消语音播报时出错:', e);
        }
        resolve(); // 超时也当作完成处理
      }, 5000);
      
      let hasResolved = false;
      
      const resolveOnce = () => {
        if (hasResolved) return;
        hasResolved = true;
        clearTimeout(timeout);
        console.log(`✅ 语音播报完成: "${text}"`);
        resolve();
      };
      
      utterance.onend = resolveOnce;
      utterance.onerror = (error) => {
        console.warn(`语音播报错误: "${text}"`, error);
        resolveOnce(); // 错误也当作完成处理，不阻塞流程
      };
      
      // 立即开始播放
      try {
        speechSynthesis.speak(utterance);
        console.log(`🎵 语音开始播放: "${text}"`);
      } catch (speakError) {
        console.warn(`语音播放启动失败: "${text}"`, speakError);
        resolveOnce();
      }
      
    } catch (error) {
      console.error('语音合成初始化错误:', error);
      resolve(); // 错误也当作完成处理
    }
  });
}

// ========== 播放提示音 "叮" ==========
function playDing() {
  return new Promise((resolve) => {
    try {
      const ding = new Audio('sounds/ding.mp3');
      
      // 设置超时机制（5秒超时）
      const timeout = setTimeout(() => {
        console.warn('音频播放超时，跳过提示音');
        resolve();
      }, 5000);
      
      // 音频加载成功
      ding.oncanplaythrough = () => {
        console.log('音频已加载，开始播放');
      };
      
      // 音频播放结束
      ding.onended = () => {
        clearTimeout(timeout);
        console.log('提示音播放完成');
        resolve();
      };
      
      // 音频播放错误
      ding.onerror = (error) => {
        clearTimeout(timeout);
        console.warn('音频播放失败，跳过提示音:', error);
        resolve(); // 音频失败也继续流程
      };
      
      // 音频加载错误
      ding.onloadstart = () => {
        console.log('开始加载音频文件...');
      };
      
      ding.onloadend = () => {
        console.log('音频文件加载完成');
      };
      
      // 尝试播放音频
      const playPromise = ding.play();
      
      if (playPromise !== undefined) {
        playPromise.then(() => {
          console.log('音频开始播放');
        }).catch((error) => {
          clearTimeout(timeout);
          console.warn('音频播放被阻止或失败，跳过提示音:', error);
          resolve(); // 播放失败也继续流程
        });
      }
      
    } catch (error) {
      console.warn('音频初始化失败，跳过提示音:', error);
      resolve(); // 任何错误都继续流程
    }
  });
}

// ========== 等待视频播放完毕 ==========
function waitForVideoEnd() {
  return new Promise((resolve) => {
    try {
      // 设置超时机制（30秒超时）
      const timeout = setTimeout(() => {
        console.warn('视频播放超时，继续下一步');
        cleanup();
        resolve();
      }, 30000);
      
      let resolved = false;
      
      function cleanup() {
        if (resolved) return;
        resolved = true;
        clearTimeout(timeout);
        previewVideo.removeEventListener('ended', endHandler);
        previewVideo.removeEventListener('error', errorHandler);
      }
      
      function endHandler() {
        console.log('视频播放完成');
        cleanup();
        resolve();
      }
      
      function errorHandler(error) {
        console.warn('视频播放错误，继续下一步:', error);
        cleanup();
        resolve(); // 视频错误也继续流程
      }
      
      // 检查视频是否已经结束
      if (previewVideo.ended) {
        console.log('视频已经结束');
        cleanup();
        resolve();
        return;
      }
      
      // 检查视频是否处于错误状态
      if (previewVideo.error) {
        console.warn('视频处于错误状态:', previewVideo.error);
        cleanup();
        resolve();
        return;
      }
      
      previewVideo.addEventListener('ended', endHandler);
      previewVideo.addEventListener('error', errorHandler);
      
    } catch (error) {
      console.warn('视频事件绑定失败:', error);
      resolve(); // 任何错误都继续流程
    }
  });
}

// 发送指令给Python服务器（临时连接方式）
function sendCommandToPython(command, data = {}) {
  return new Promise((resolve, reject) => {
    try {
      console.log(`🔄 创建临时连接发送指令: ${command}`);
      
      // 添加UI提示
      const oldTitle = taskTitleElem.innerText;
      taskTitleElem.innerText = `正在连接服务器...`;
      
      const tempWs = new WebSocket('ws://localhost:8765');
      
      // 设置连接超时
      const connectionTimeout = setTimeout(() => {
        console.error('WebSocket连接超时');
        taskTitleElem.innerText = `服务器连接超时，请确认服务器已启动`;
        setTimeout(() => {
          taskTitleElem.innerText = oldTitle;
        }, 3000);
        reject(new Error('连接超时'));
      }, 3000);
      
      tempWs.onopen = () => {
        clearTimeout(connectionTimeout);
        const message = {
          command: command,
          ...data
        };
        tempWs.send(JSON.stringify(message));
        console.log('✅ 指令发送成功:', command, data);
        taskTitleElem.innerText = oldTitle;
        
        // 发送完立即关闭连接
        setTimeout(() => {
          tempWs.close();
          resolve();
        }, 50);
      };
      
      tempWs.onerror = (error) => {
        clearTimeout(connectionTimeout);
        console.warn('临时连接失败:', error);
        taskTitleElem.innerText = `服务器连接失败，请确认Python服务已启动`;
        
        // 3秒后恢复原标题
        setTimeout(() => {
          taskTitleElem.innerText = oldTitle;
        }, 3000);
        
        reject(error);
      };
      
      tempWs.onclose = () => {
        clearTimeout(connectionTimeout);
        console.log('📴 临时连接已关闭');
      };
      
    } catch (error) {
      console.error('创建临时连接时出错:', error);
      taskTitleElem.innerText = `连接服务器失败: ${error.message}`;
      
      // 3秒后恢复原标题
      setTimeout(() => {
        taskTitleElem.innerText = oldTitle;
      }, 3000);
      
      reject(error);
    }
  });
}