import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import vision from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0';

const { FaceLandmarker, FilesetResolver } = vision;

const blendshapesMap = {
    // '_neutral': '',
    'browDownLeft': 'browDown_L',
    'browDownRight': 'browDown_R',
    'browInnerUp': 'browInnerUp',
    'browOuterUpLeft': 'browOuterUp_L',
    'browOuterUpRight': 'browOuterUp_R',
    'cheekPuff': 'cheekPuff',
    'cheekSquintLeft': 'cheekSquint_L',
    'cheekSquintRight': 'cheekSquint_R',
    'eyeBlinkLeft': 'eyeBlink_L',
    'eyeBlinkRight': 'eyeBlink_R',
    'eyeLookDownLeft': 'eyeLookDown_L',
    'eyeLookDownRight': 'eyeLookDown_R',
    'eyeLookInLeft': 'eyeLookIn_L',
    'eyeLookInRight': 'eyeLookIn_R',
    'eyeLookOutLeft': 'eyeLookOut_L',
    'eyeLookOutRight': 'eyeLookOut_R',
    'eyeLookUpLeft': 'eyeLookUp_L',
    'eyeLookUpRight': 'eyeLookUp_R',
    'eyeSquintLeft': 'eyeSquint_L',
    'eyeSquintRight': 'eyeSquint_R',
    'eyeWideLeft': 'eyeWide_L',
    'eyeWideRight': 'eyeWide_R',
    'jawForward': 'jawForward',
    'jawLeft': 'jawLeft',
    'jawOpen': 'jawOpen',
    'jawRight': 'jawRight',
    'mouthClose': 'mouthClose',
    'mouthDimpleLeft': 'mouthDimple_L',
    'mouthDimpleRight': 'mouthDimple_R',
    'mouthFrownLeft': 'mouthFrown_L',
    'mouthFrownRight': 'mouthFrown_R',
    'mouthFunnel': 'mouthFunnel',
    'mouthLeft': 'mouthLeft',
    'mouthLowerDownLeft': 'mouthLowerDown_L',
    'mouthLowerDownRight': 'mouthLowerDown_R',
    'mouthPressLeft': 'mouthPress_L',
    'mouthPressRight': 'mouthPress_R',
    'mouthPucker': 'mouthPucker',
    'mouthRight': 'mouthRight',
    'mouthRollLower': 'mouthRollLower',
    'mouthRollUpper': 'mouthRollUpper',
    'mouthShrugLower': 'mouthShrugLower',
    'mouthShrugUpper': 'mouthShrugUpper',
    'mouthSmileLeft': 'mouthSmile_L',
    'mouthSmileRight': 'mouthSmile_R',
    'mouthStretchLeft': 'mouthStretch_L',
    'mouthStretchRight': 'mouthStretch_R',
    'mouthUpperUpLeft': 'mouthUpperUp_L',
    'mouthUpperUpRight': 'mouthUpperUp_R',
    'noseSneerLeft': 'noseSneer_L',
    'noseSneerRight': 'noseSneer_R',
    // '': 'tongueOut'
};

export class FaceCapture {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.face = null;
        this.eyeL = null;
        this.eyeR = null;
        this.eyeRotationLimit = THREE.MathUtils.degToRad(30);
        this.transform = new THREE.Object3D();
        
        console.log('FaceCapture构造函数开始，容器ID:', containerId);
        console.log('容器元素:', this.container);
        
        this.init();
    }

    async init() {
        try {
            console.log('开始初始化面部捕捉...');
            
            // 显示加载提示
            this.container.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #007BFF;">正在初始化面部捕捉...</div>';
            
            // 初始化渲染器
            console.log('初始化渲染器...');
            this.renderer = new THREE.WebGLRenderer({ antialias: true });
            this.renderer.setPixelRatio(window.devicePixelRatio);
            this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
            this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
            
            // 清空容器并添加渲染器
            this.container.innerHTML = '';
            this.container.appendChild(this.renderer.domElement);

            // 设置场景和相机
            console.log('设置场景和相机...');
            this.setupScene();
            
            // 加载模型和设置WebCam
            console.log('开始加载模型和设置WebCam...');
            await Promise.all([
                this.loadModel(),
                this.setupWebcam(),
                this.setupMediaPipe()
            ]);

            console.log('所有初始化完成，开始动画循环...');
            // 开始动画循环
            this.animate();

            // 添加窗口大小调整监听
            window.addEventListener('resize', () => this.onWindowResize());
            
            console.log('面部捕捉初始化成功！');
            
        } catch (error) {
            console.error('FaceCapture初始化失败:', error);
            this.container.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: #dc3545; text-align: center; padding: 20px;">
                    <div style="font-size: 16px; margin-bottom: 10px;">⚠️ 面部捕捉初始化失败</div>
                    <div style="font-size: 12px; color: #666;">${error.message}</div>
                    <div style="font-size: 10px; color: #999; margin-top: 10px;">请检查浏览器控制台获取详细错误信息</div>
                </div>
            `;
        }
    }

    setupScene() {
        console.log('开始设置场景...');
        this.camera = new THREE.PerspectiveCamera(60, this.container.clientWidth / this.container.clientHeight, 1, 100);
        this.camera.position.z = 5;
        this.camera.position.y = 0;
        this.camera.lookAt(0, 0, 0);

        this.scene = new THREE.Scene();
        this.scene.scale.x = -1;

        const environment = new RoomEnvironment();
        const pmremGenerator = new THREE.PMREMGenerator(this.renderer);

        this.scene.background = new THREE.Color(0x666666);
        this.scene.environment = pmremGenerator.fromScene(environment).texture;
        console.log('场景设置完成');
    }

    async loadModel() {
        console.log('开始加载3D模型...');
        const ktx2Loader = new KTX2Loader()
            .setTranscoderPath('./libs/examples/libs/basis/')
            .detectSupport(this.renderer);

        const textureLoader = new THREE.TextureLoader();
        console.log('开始加载纹理...');
        
        // 创建一个简单的纹理作为后备选项
        let roughnessMap = null;
        try {
            roughnessMap = textureLoader.load('./res.png', 
            () => console.log('纹理加载成功'),
            undefined,
                (error) => {
                    console.warn('纹理加载失败，使用默认材质:', error);
                    // 不阻止模型加载，继续使用默认材质
                }
        );
        } catch (error) {
            console.warn('纹理初始化失败，将使用默认材质:', error);
        }

        return new Promise((resolve, reject) => {
            console.log('开始加载GLTF模型...');
            new GLTFLoader()
                .setKTX2Loader(ktx2Loader)
                .setMeshoptDecoder(MeshoptDecoder)
                .load('./man.gltf', (gltf) => {
                    console.log('GLTF模型加载成功:', gltf.scene);
                    const mesh = gltf.scene;
                    mesh.scale.set(2, 2, 2); // 放大2倍
                    this.scene.add(mesh);

                    console.log('--- 模型中的可设置材质的对象列表 ---');
                    let meshCount = 0;
                    mesh.traverse((child) => {
                        if (child.isMesh) {
                            meshCount++;
                            console.log(`${meshCount}. 网格名称: "${child.name}"`);
                            console.log(`   - 类型: ${child.type}`);
                            console.log(`   - 是否可见: ${child.visible}`);
                            if (child.material) {
                                console.log(`   - 当前材质类型: ${child.material.type}`);
                            }
                            console.log('-------------------');
                        }
                    });
                    console.log(`总共找到 ${meshCount} 个可设置材质的对象`);

                    this.setupMaterials(mesh, roughnessMap);
                    console.log('模型材质设置完成');
                    resolve();
                }, undefined, (error) => {
                    console.error('GLTF模型加载失败:', error);
                    reject(error);
                });
        });
    }

    setupMaterials(mesh, roughnessMap) {
        mesh.traverse((child) => {
            if (child.isMesh) {
                // 创建白色材质（用于眼睛和牙齿）
                const whiteMaterial = new THREE.MeshStandardMaterial({
                    color: 0xffffff,    // 白色
                    roughness: 0.2,      // 较光滑
                    envMapIntensity: 1.0 // 较强的环境反射
                });

                // 创建肤色材质（用于皮肤）
                const skinMaterialOptions = {
                    color: 0xffdbac,    // 肤色
                    roughness: 0.8,      // 基础粗糙度
                    envMapIntensity: 0.5 // 环境反射强度
                };
                
                // 只有在纹理成功加载时才添加粗糙度贴图
                if (roughnessMap && roughnessMap.image) {
                    skinMaterialOptions.roughnessMap = roughnessMap;
                }
                
                const skinMaterial = new THREE.MeshStandardMaterial(skinMaterialOptions);

                // 打印当前处理的网格名称
                console.log('正在处理网格:', child.name);

                // 根据网格名称设置不同的材质
                if (child.name === 'eyeLeft' || 
                    child.name === 'eyeRight' || 
                    child.name.toLowerCase().includes('teeth')) {
                    console.log('设置白色材质给:', child.name);
                    child.material = whiteMaterial;
                } else {
                    console.log('设置肤色材质给:', child.name);
                    child.material = skinMaterial;
                }
                
                // 如果这个网格是头部，设置为face
                if (child.name.toLowerCase().includes('head')) {
                    this.face = child;
                }
                // 如果这个网格是眼睛，设置为eyeL或eyeR
                else if (child.name.toLowerCase().includes('eye')) {
                    if (child.name.toLowerCase().includes('left')) {
                        this.eyeL = child;
                    } else if (child.name.toLowerCase().includes('right')) {
                        this.eyeR = child;
                    }
                }
            }
        });

        // 如果找不到特定的网格，使用默认值
        if (!this.face) {
            console.warn('No head mesh found in the model');
            this.face = mesh.children[0];  // 使用第一个子网格作为face
        }

        // GUI (可选)
        // const gui = new GUI();
        // gui.close();
        // if (this.face && this.face.morphTargetInfluences) {
        //     const influences = this.face.morphTargetInfluences;
        //     for (const [key, value] of Object.entries(this.face.morphTargetDictionary || {})) {
        //         gui.add(influences, value, 0, 1, 0.01)
        //             .name(key.replace('blendShape1.', ''))
        //             .listen(influences);
        //     }
        // }
    }

    async setupWebcam() {
        console.log('开始设置网络摄像头...');
        this.video = document.createElement('video');
        this.videoTexture = new THREE.VideoTexture(this.video);
        this.videoTexture.colorSpace = THREE.SRGBColorSpace;

        // 创建一个不可见的视频平面
        const geometry = new THREE.PlaneGeometry(1, 1);
        const material = new THREE.MeshBasicMaterial({ map: this.videoTexture, depthWrite: false, visible: false });
        this.videomesh = new THREE.Mesh(geometry, material);
        // 将视频平面放置在场景外
        this.videomesh.position.z = -1000;
        this.scene.add(this.videomesh);

        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            try {
                console.log('请求摄像头权限...');
                const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
                this.video.srcObject = stream;
                await this.video.play();
                console.log('摄像头设置成功');
            } catch (error) {
                console.error('无法访问摄像头:', error);
                throw new Error(`摄像头访问失败: ${error.message}`);
            }
        } else {
            const errorMsg = '浏览器不支持摄像头访问';
            console.error(errorMsg);
            throw new Error(errorMsg);
        }
    }

    async setupMediaPipe() {
        console.log('开始设置MediaPipe...');
        try {
            const filesetResolver = await FilesetResolver.forVisionTasks(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
            );
            console.log('MediaPipe文件集解析器设置完成');

            this.faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
                baseOptions: {
                    modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                    delegate: 'GPU'
                },
                outputFaceBlendshapes: true,
                outputFacialTransformationMatrixes: true,
                runningMode: 'VIDEO',
                numFaces: 1
            });
            console.log('MediaPipe面部标记器创建成功');
        } catch (error) {
            console.error('MediaPipe设置失败:', error);
            throw new Error(`MediaPipe初始化失败: ${error.message}`);
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        
        if (this.video && this.video.readyState >= HTMLMediaElement.HAVE_METADATA && this.faceLandmarker) {
            try {
                const results = this.faceLandmarker.detectForVideo(this.video, Date.now());
                this.updateFaceTransform(results);
                this.updateBlendShapes(results);
            } catch (error) {
                console.warn('面部检测失败:', error);
            }
        }

        if (this.videomesh && this.video) {
            this.videomesh.scale.x = this.video.videoWidth / 100;
            this.videomesh.scale.y = this.video.videoHeight / 100;
        }

        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }

    updateFaceTransform(results) {
        if (results.facialTransformationMatrixes.length > 0) {
            const facialTransformationMatrixes = results.facialTransformationMatrixes[0].data;

            this.transform.matrix.fromArray(facialTransformationMatrixes);
            this.transform.matrix.decompose(this.transform.position, this.transform.quaternion, this.transform.scale);

            const object = this.scene.getObjectByName('grp_transform');

            if (object) {
                object.position.x = this.transform.position.x;
                object.position.y = this.transform.position.z + 40;
                object.position.z = -this.transform.position.y;

                // 保持正面朝向摄像机
                object.rotation.set(0, 0, 0);
            }
        }
    }

    updateBlendShapes(results) {
        if (results.faceBlendshapes.length > 0) {
            const faceBlendshapes = results.faceBlendshapes[0].categories;

            // Morph values does not exist on the eye meshes, so we map the eyes blendshape score into rotation values
            const eyeScore = {
                leftHorizontal: 0,
                rightHorizontal: 0,
                leftVertical: 0,
                rightVertical: 0,
            };

            for (const blendshape of faceBlendshapes) {
                const categoryName = blendshape.categoryName;
                const score = blendshape.score;

                if (this.face && this.face.morphTargetDictionary) {
                    const index = this.face.morphTargetDictionary[blendshapesMap[categoryName]];

                    if (index !== undefined) {
                        this.face.morphTargetInfluences[index] = score;
                    }
                }

                // There are two blendshape for movement on each axis (up/down , in/out)
                // Add one and subtract the other to get the final score in -1 to 1 range
                switch (categoryName) {
                    case 'eyeLookInLeft':
                        eyeScore.leftHorizontal += score;
                        break;
                    case 'eyeLookOutLeft':
                        eyeScore.leftHorizontal -= score;
                        break;
                    case 'eyeLookInRight':
                        eyeScore.rightHorizontal -= score;
                        break;
                    case 'eyeLookOutRight':
                        eyeScore.rightHorizontal += score;
                        break;
                    case 'eyeLookUpLeft':
                        eyeScore.leftVertical -= score;
                        break;
                    case 'eyeLookDownLeft':
                        eyeScore.leftVertical += score;
                        break;
                    case 'eyeLookUpRight':
                        eyeScore.rightVertical -= score;
                        break;
                    case 'eyeLookDownRight':
                        eyeScore.rightVertical += score;
                        break;
                }
            }

            if (this.eyeL) {
                this.eyeL.rotation.z = eyeScore.leftHorizontal * this.eyeRotationLimit;
                this.eyeL.rotation.x = eyeScore.leftVertical * this.eyeRotationLimit;
            }
            if (this.eyeR) {
                this.eyeR.rotation.z = eyeScore.rightHorizontal * this.eyeRotationLimit;
                this.eyeR.rotation.x = eyeScore.rightVertical * this.eyeRotationLimit;
            }
        }
    }

    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
} 