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
    with st.sidebar:
        st.header("🤿 潜水备忘录")

        if not st.session_state.onboarding_complete:
            progress = (st.session_state.onboarding_step - 1) / 3
            st.write("正在建立连接，请在对话框完成初始化...")
            st.progress(progress)
            st.caption(f"进度：{int(progress * 100)}%")
        else:
            st.success("✅ Buddy 已记下你的档案")
            profile = st.session_state.user_profile
            st.write(f"**等级：** {profile['level']}")
            st.write(f"**经验：** {profile['logs']}")

            st.write("**偏好标签：**")
            for i, tag in enumerate(profile['preference']):
                cols = st.columns([4, 1])
                cols[0].caption(f"• {tag}")
                if cols[1].button("❌", key=f"del_{i}"):
                    st.session_state.user_profile['preference'].pop(i)
                    with open(user_file, 'w', encoding='utf-8') as f:
                        json.dump(st.session_state.user_profile, f, ensure_ascii=False, indent=4)
                    st.rerun()

        # 档案详情展示
        profile = st.session_state.user_profile
        st.subheader("🗺️ 足迹")
        sites = profile.get("visited_sites", [])
        st.write(" ".join([f"`{s}`" for s in sites]) if sites else "还没留下足迹")

        st.subheader("🐬 生物集邮")
        animals = profile.get("seen_animals", [])
        st.write(" ".join([f"`{a}`" for a in animals]) if animals else "还没集邮")

        st.subheader("🧠 教练笔记")
        notes = profile.get("dynamic_notes", []) + profile.get("dive_tips", [])
        if notes:
            for n in notes:
                st.info(n)