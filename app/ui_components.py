# ui_components.py
import streamlit as st
import json


from utils import parse_trip_content

def render_super_trip_card(doc, idx):
    """15个字段的超级卡片：结构化、图标化、层级分明"""
    # 调用 utils 中的解析函数
    details = parse_trip_content(doc.page_content)
    meta = doc.metadata

    with st.container(border=True):
        # --- 第一层：头部 (名称与价格) ---
        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            st.markdown(f"### 🚢 {meta.get('boat_nameEN', '未知')}{meta.get('boat_nameCN', '')}")
            st.caption(f" | 📍 {meta.get('locationName', '未知')}")
            st.caption(f" | 路线：{meta.get('tour_nameCN', '未知')}({meta.get('tour_nameEN', '')})")
            st.caption(f" | 出发起点：{meta.get('departureLocation', '未知')}  返程终点：{meta.get('arrivalLocation', '未知')}")
        with col_h2:
            st.markdown(f"### :orange[{details['price_str']}]")
            st.caption(f"🔥 余位: {details['available_count']}")

        st.divider()

        # --- 第二层：核心规格 (4列并排) ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.write("⏱️ **时长**")
            st.write(f"{meta.get('duration', '未知')}天{meta.get('nights', '未知')}晚")
        with c2:
            st.write("🤿 **潜水**")
            st.write(f"{meta.get('dives', '未知')}次")
        with c3:
            st.write("🎖️ **要求**")
            st.write(f"{meta.get('certification', 'AOW')}")
        with c4:
            st.write("📊 **经验**")
            st.write(f"{meta.get('experience', 50)} 瓶")

        # --- 第三层：服务设施 ---
        nitrox_icon = "✅" if details['nitrox'] == "是" else "❌"
        wifi_icon = "✅" if details['wifi'] == "是" else "❌"
        tech_icon = "✅" if details['tech_friendly'] == "是" else "❌"
        st.markdown(f"高氧: {nitrox_icon} | Wi-Fi: {wifi_icon} | 技术潜水友好: {tech_icon}")

        # --- 第四层：日期信息 ---
        st.info(f"📅 **航期：** {meta.get('departureDate_display', '未知')}  ➡️  {meta.get('arrivalDate_display', '未知')}")

        # --- 第五层：政策与外链 ---
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            with st.expander("📝 预订与取消政策 (查看详情)"):
                st.write("**预订政策:**")
                st.caption(details['booking_policy'])
                st.write("**取消政策:**")
                st.caption(details['cancellation_policy'])
        with col_f2:
            source_url = meta.get("Metadata_source", "#")
            st.link_button("🔗 查看原站详情", source_url, use_container_width=True)

def display_trip_results(docs):
    """负责从保险箱中提取文档、去重，并调用渲染函数"""
    if not docs:
        return

    st.write("---")
    st.caption("📑 匹配到的船宿详细方案")

    seen_boat_ids = set()
    display_count = 0

    for doc in docs:
        if doc.metadata.get("category") != "船宿行程":
            continue

        boat_id = doc.metadata.get("boatId")
        if boat_id in seen_boat_ids:
            continue

        seen_boat_ids.add(boat_id)
        render_super_trip_card(doc, display_count)
        display_count += 1
        if display_count >= 5:
            break

def render_sidebar(user_file):
    """侧边栏用户画像与管理界面"""
    import state_manager
    import json

    with st.sidebar:
        st.header("🤿 潜水备忘录")

        # --- 情况 A：如果是新用户，正在做问卷 ---
        if not st.session_state.get("onboarding_complete", False):
            st.info("正在建立连接...")
            step = st.session_state.get("onboarding_step", 1)
            progress = (step - 1) / 3
            st.progress(progress)
            st.caption("完成初始化后即可开启管理模式")

        # --- 情况 B：已经是老用户 ---
        else:
            # 1. 给 Toggle 增加唯一 key
            edit_toggle = st.toggle("🛠️ 进入档案管理模式",
                                    value=st.session_state.get("edit_mode", False),
                                    key="sidebar_edit_mode_toggle")  # 👈 必须加 key
            st.session_state.edit_mode = edit_toggle

            profile = st.session_state.user_profile
            st.divider()

            if not st.session_state.edit_mode:
                # --- 【展示模式】 ---
                st.write(f"**当前等级：** `{profile.get('level')}`")
                st.write(f"**潜水瓶数：** `{profile.get('logs')}`")
                st.write("**偏好标签：**")
                for i, tag in enumerate(profile['preference']):
                    cols = st.columns([4, 1])
                    cols[0].caption(f"• {tag}")

                # 档案详情展示
                profile = st.session_state.user_profile
                st.subheader("🗺️ 足迹")
                sites = profile.get("visited_sites", [])
                st.write(" ".join([f"`{s}`" for s in sites]) if sites else "还没留下足迹")

                st.subheader("🐬 生物集邮")
                animals = profile.get("seen_animals", [])
                st.write(" ".join([f"`{a}`" for a in animals]) if animals else "还没集邮")

                st.subheader("⭐ buddy笔记")
                notes = profile.get("dynamic_notes", []) + profile.get("dive_tips", [])
                if notes:
                    for n in notes:
                        st.info(n)
            else:
                # --- 【管理模式】 ---
                st.info("💡 修改将实时保存至本地 JSON。")

                # A. 等级修改 - 增加唯一 key
                level_options = ["初学者", "OW", "AOW及以上"]
                try:
                    default_idx = level_options.index(profile.get('level', 'OW'))
                except:
                    default_idx = 1

                new_level = st.selectbox("修改等级",
                                         options=level_options,
                                         index=default_idx,
                                         key="sb_level_select")  # 👈 必须加 key

                # B. 瓶数修改 - 增加唯一 key
                new_logs = st.text_input("修改瓶数",
                                         value=str(profile.get('logs', '0-20')),
                                         key="sb_logs_input")  # 👈 必须加 key

                # 检查并保存基础信息
                if new_level != profile['level'] or new_logs != profile['logs']:
                    profile['level'] = new_level
                    profile['logs'] = new_logs
                    state_manager.save_user_profile(user_file, profile)
                    st.toast("✅ 基础档案已更新")

                # 偏好标签 - 增加唯一 key
                st.subheader("🎯 偏好标签")
                pref_options = ["看大货 (鲨鱼/Manta)", "找微距 (海兔)", "放流潜水", "沉船/洞穴", "水下摄影", "夜潜"]
                current_prefs = [p for p in profile.get('preference', []) if p in pref_options]
                # 使用 multiselect 进行管理
                new_prefs = st.multiselect(
                    "调整我的偏好",
                    options=pref_options,
                    default=current_prefs,
                    key="sb_pref_multiselect"  # 必须加唯一 key
                )
                # 检查是否有变动并保存
                if new_prefs != profile.get('preference'):
                    profile['preference'] = new_prefs
                    state_manager.save_user_profile(user_file, profile)
                    st.toast("✅ 偏好设置已更新")

                # 足迹管理 - 给删除按钮增加带索引的 key
                st.subheader("🗺️ 足迹")
                sites = profile.get("visited_sites", [])
                for i, site in enumerate(sites):
                    c1, c2 = st.columns([4, 1])
                    c1.caption(site)
                    # 使用 f-string 确保每个按钮的 key 都是唯一的
                    if c2.button("❌", key=f"btn_del_site_{i}"):
                        profile["visited_sites"].pop(i)
                        state_manager.save_user_profile(user_file, profile)
                        st.rerun()
                # 新增足迹 - 增加唯一 key
                new_site = st.text_input("➕ 新增足迹", key="sb_add_site_input")
                if st.button("确认添加", key="sb_add_site_btn"):
                    if new_site:
                        if "visited_sites" not in profile: profile["visited_sites"] = []
                        profile["visited_sites"].append(new_site.strip())
                        state_manager.save_user_profile(user_file, profile)
                        st.rerun()

                # 生物集邮 - 给删除按钮增加带索引的 key
                st.subheader("🗺 生物集邮")
                animals = profile.get("seen_animals", [])
                for i, animal in enumerate(animals):
                    c1, c2 = st.columns([4, 1])
                    c1.caption(animal)
                    # 使用 f-string 确保每个按钮的 key 都是唯一的
                    if c2.button("❌", key=f"btn_del_animal_{i}"):
                        profile["seen_animals"].pop(i)
                        state_manager.save_user_profile(user_file, profile)
                        st.rerun()
                # 新增生物集邮 - 增加唯一 key
                new_animal = st.text_input("➕ 新增生物", key="sb_add_animal_input")
                if st.button("确认添加", key="sb_add_animal_btn"):
                    if new_animal:
                        if "seen_animals" not in profile: profile["seen_animals"] = []
                        profile["seen_animals"].append(new_animal.strip())
                        state_manager.save_user_profile(user_file, profile)
                        st.rerun()

